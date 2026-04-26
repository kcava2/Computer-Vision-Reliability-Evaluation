"""
Model Reliability Score (MRS) framework.

Reframes misclassification as a failure event and applies reliability
engineering principles across five components:
  P_s   — Probability Score   (baseline failure rate)
  DS    — Durability Score     (degradation under distribution shift)
  DepS* — Dependability Score  (calibration-relative CVaR of failure confidence)
  QoTS  — Quality over Time    (lag-1 failure autocorrelation)
  AS    — Availability Score   (MTBF / (MTBF + MTTR) on perturbed data)
  MRS   — weighted harmonic mean of all five
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, t as t_dist


_DEFAULT_CONFIG = {
    'w_I': 0.5,           # Type I (false positive) severity weight
    'w_II': 0.5,          # Type II (false negative) severity weight
    'alpha': 0.95,        # CVaR confidence level for DepS*
    'ece_bins': 15,       # number of equal-width bins for ECE
    'sigma': 0.25,         # Gaussian noise sigma used in D_tilde (reference)
    'weights_equal': [0.2, 0.2, 0.2, 0.2, 0.2],
    'bootstrap_n': 1000,
}


class ReliabilityEvaluator:
    """Evaluates a PyTorch model using reliability engineering metrics.

    Args:
        model:  Trained nn.Module that outputs raw logits (B, num_classes).
        device: torch.device for inference.
        config: Optional dict to override default hyperparameters.
    """

    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self._results = {}

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _run_inference(self, loader):
        """Run full inference pass and collect per-sample failure information.

        Returns a dict with all arrays needed for metric computation.
        Every misclassification is simultaneously a Type I FP for the predicted
        class and a Type II FN for the true class.
        w(tau_n) = w_I + w_II — with equal defaults (0.5 + 0.5 = 1.0) this yields
        P(F|D) = K/N, matching raw accuracy (P_s = 1 - error_rate).
        To penalise one error type more, increase its weight; the sum sets the
        overall severity multiplier.
        """
        w_avg = self.config['w_I'] + self.config['w_II']

        all_preds = []
        all_labels = []
        all_confidences = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)                     # (B, C)
                probs = F.softmax(logits, dim=1)                # (B, C)
                preds = logits.argmax(dim=1)                    # (B,)
                confs = probs.max(dim=1).values                 # (B,)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_confidences.extend(confs.cpu().tolist())

        all_preds = np.array(all_preds, dtype=np.int64)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_confidences = np.array(all_confidences, dtype=np.float64)

        failure_mask = all_preds != all_labels                  # bool[N]
        failure_indices = np.where(failure_mask)[0]             # int[K]
        failure_confidences = all_confidences[failure_mask]     # float[K]

        N = len(all_preds)
        K = int(failure_mask.sum())
        # P(F|D) weighted by average error-type severity
        pf = (K / N) * w_avg if N > 0 else 0.0

        consecutive_runs = self._compute_consecutive_runs(failure_indices)

        return {
            'N': N,
            'K': K,
            'pf': pf,
            'c_bar': float(np.mean(all_confidences)),  # global mean confidence over all N inferences
            'failure_mask': failure_mask,
            'all_confidences': all_confidences,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'failure_indices': failure_indices,
            'failure_confidences': failure_confidences,
            'consecutive_failure_runs': consecutive_runs,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_consecutive_runs(self, failure_indices):
        """Return list of consecutive failure run lengths."""
        if len(failure_indices) == 0:
            return []
        runs = []
        run_len = 1
        for i in range(1, len(failure_indices)):
            if failure_indices[i] == failure_indices[i - 1] + 1:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        return runs

    def _compute_cvar(self, confidences):
        """CVaR at config['alpha'] on an array of (possibly calibration-relative) scores."""
        alpha = self.config['alpha']
        if len(confidences) == 0:
            return 0.0
        sorted_c = np.sort(confidences)
        n = len(sorted_c)
        cutoff_idx = int(np.floor(alpha * n))
        cutoff_idx = min(cutoff_idx, n - 1)
        tail = sorted_c[cutoff_idx:]
        return float(np.mean(tail))

    def _compute_ece(self, all_confidences, failure_mask, n_bins=None):
        """Expected Calibration Error over equal-width confidence bins.

        ECE = sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|
        """
        n_bins = n_bins or self.config['ece_bins']
        N = len(all_confidences)
        correct_mask = ~failure_mask
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            in_bin = (all_confidences >= lo) & (all_confidences < hi)
            if i == n_bins - 1:  # include upper boundary on last bin
                in_bin = (all_confidences >= lo) & (all_confidences <= hi)
            n_b = int(in_bin.sum())
            if n_b == 0:
                continue
            acc_b = float(correct_mask[in_bin].mean())
            conf_b = float(all_confidences[in_bin].mean())
            ece += (n_b / N) * abs(acc_b - conf_b)
        return float(ece)

    def _rebuild_result(self, result, idx):
        """Build a bootstrapped result dict by resampling idx from result arrays."""
        w_avg = self.config['w_I'] + self.config['w_II']
        N = result['N']
        fm = result['failure_mask'][idx]
        ac = result['all_confidences'][idx]
        fi = np.where(fm)[0]
        fc = ac[fm]
        K = int(fm.sum())
        pf = (K / N) * w_avg if N > 0 else 0.0
        runs = self._compute_consecutive_runs(fi)
        return {
            'N': N, 'K': K, 'pf': pf,
            'c_bar': float(np.mean(ac)),  # mean confidence of this bootstrap sample
            'failure_mask': fm,
            'all_confidences': ac,
            'failure_indices': fi,
            'failure_confidences': fc,
            'consecutive_failure_runs': runs,
        }

    # ------------------------------------------------------------------
    # Metric 1 — Probability Score
    # ------------------------------------------------------------------

    def compute_probability_score(self, r_D):
        """P_s = 1 - P(F|D).  Higher means fewer weighted failures on clean data."""
        return float(np.clip(1.0 - r_D['pf'], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 2 — Durability Score
    # ------------------------------------------------------------------

    def compute_durability_score(self, r_D, r_Dprime):
        """DS = clip(1 - (P(F|D') - P(F|D)) / P(F|D), 0, 1).

        DS = 1 means no degradation under shift; DS = 0 means total collapse.
        """
        pf_D = r_D['pf']
        pf_Dp = r_Dprime['pf']
        if pf_D == 0.0:
            return 1.0  # no baseline failures to degrade from
        ds = 1.0 - (pf_Dp - pf_D) / pf_D
        return float(np.clip(ds, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 3 — Dependability Score
    # ------------------------------------------------------------------

    def compute_dependability_score(self, r_D):
        """DepS* = calibration-adjusted dependability using relative failure confidence.

        Uses calibration-relative confidence c_n* = c_n - c_bar at each failure,
        where c_bar is the global mean confidence over all N inferences.
        This separates failure-specific overconfidence from systemic miscalibration.

        DepS* = 1 - (clip(CVaR_alpha(F*), -1, 1) / 2 + 0.5)
          CVaR = +1  (failures far above average confidence) -> DepS* = 0.0
          CVaR =  0  (failures at average confidence)        -> DepS* = 0.5
          CVaR = -1  (failures below average confidence)     -> DepS* = 1.0

        Edge case: no failures -> DepS* = 1.0.
        """
        if r_D['K'] == 0:
            return 1.0
        c_bar = r_D['c_bar']
        rel_confs = r_D['failure_confidences'] - c_bar  # calibration-relative F*
        cvar = self._compute_cvar(rel_confs)
        deps = 1.0 - (float(np.clip(cvar, -1.0, 1.0)) / 2.0 + 0.5)
        return float(np.clip(deps, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 4 — Quality over Time Score (failure autocorrelation)
    # ------------------------------------------------------------------

    def compute_qots(self, r_D):
        """QoTS via lag-1 autocorrelation of the binary failure sequence.

        Replaces Weibull analysis, which failed goodness-of-fit on CIFAR-100
        (Anderson-Darling stat=273.74 >> crit=0.755) due to frequent, compressed
        inter-failure intervals that violate Weibull's rare-event assumption.

        X_n = 1[y_hat_n != y_n] — binary failure indicator, length N.
        rho = corr(X[:-1], X[1:]) — lag-1 autocorrelation.
        QoTS = 1 - |rho|

        rho ~ 0: failures are memoryless/random (best case, QoTS ~ 1)
        rho > 0: failures cluster together, wear-out-like pattern
        rho < 0: failures alternate with successes in a pattern
        |rho| ~ 1: highly systematic, QoTS ~ 0

        Returns dict with score, rho, t_stat, p_value, and pattern diagnosis.
        """
        X = r_D['failure_mask'].astype(np.float64)
        N = len(X)

        base = {
            'score': 0.5, 'rho': None, 't_stat': None,
            'p_value': None, 'significant_at_0.05': None,
            'pattern': 'Insufficient data for autocorrelation analysis',
        }

        if N < 3:
            return base

        rho = float(np.corrcoef(X[:-1], X[1:])[0, 1])

        if np.isnan(rho):
            warnings.warn('QoTS: corrcoef returned NaN (likely constant failure sequence).', RuntimeWarning)
            return base

        qots = float(np.clip(1.0 - abs(rho), 0.0, 1.0))

        # t-test: rho = 0 under null, t ~ t(N-2)
        if abs(rho) < 1.0:
            t_stat = float(rho * np.sqrt(N - 2) / np.sqrt(max(1.0 - rho ** 2, 1e-15)))
            p_value = float(t_dist.sf(abs(t_stat), df=N - 2) * 2)
        else:
            t_stat = float('inf') * np.sign(rho)
            p_value = 0.0

        significant = bool(p_value <= 0.05)

        if abs(rho) < 0.1 and not significant:
            pattern = 'Random — failures are unpatterned and memoryless'
        elif rho > 0.1 and significant:
            pattern = 'Clustered — systematic wear-out-like failure behavior'
        elif rho < -0.1 and significant:
            pattern = 'Alternating — systematic oscillating failure pattern'
        else:
            pattern = 'Weak pattern — insufficient evidence of systematic failures'

        return {
            'score': qots,
            'rho': rho,
            't_stat': t_stat,
            'p_value': p_value,
            'significant_at_0.05': significant,
            'pattern': pattern,
        }

    # ------------------------------------------------------------------
    # Metric 5 — Availability Score
    # ------------------------------------------------------------------

    def compute_availability_score(self, r_Dtilde):
        """AS = MTBF_ML / (MTBF_ML + MTTR_ML) on perturbed data.

        MTBF_ML = mean inferences between failure runs.
        MTTR_ML = mean length of consecutive failure runs.
        """
        N = r_Dtilde['N']
        K = r_Dtilde['K']
        failure_indices = r_Dtilde['failure_indices']
        runs = r_Dtilde['consecutive_failure_runs']

        if K == 0:
            return 1.0
        if K == N:
            return 0.0

        # MTBF: mean gap between separate failure runs (gaps with >=1 success)
        intervals = np.diff(failure_indices).astype(np.float64)
        success_gaps = intervals[intervals > 1]
        if len(success_gaps) > 0:
            mtbf = float(np.mean(success_gaps))
        else:
            # All failures are in one contiguous run; treat as N total inferences
            mtbf = float(N)

        mttr = float(np.mean(runs)) if runs else 0.0

        if mttr == 0.0:
            return 1.0

        return float(np.clip(mtbf / (mtbf + mttr), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Composite MRS
    # ------------------------------------------------------------------

    def compute_mrs(self, scores, weights):
        """Weighted harmonic mean: sum(w) / sum(w_i / score_i).

        The harmonic mean ensures a single catastrophically low sub-score
        significantly drags down the composite.
        """
        w = np.array(weights, dtype=np.float64)
        s = np.array([max(sc, 1e-9) for sc in scores], dtype=np.float64)
        return float(np.sum(w) / np.sum(w / s))

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_confidence_interval(self, result, metric_fn, n=None):
        """Resample N samples with replacement and recompute metric_fn 1000 times.

        metric_fn must accept a result dict (not a loader) so inference is not
        repeated on each bootstrap iteration.
        """
        n = n or self.config['bootstrap_n']
        N = result['N']
        rng = np.random.default_rng(42)
        bs_scores = []
        for _ in range(n):
            idx = rng.integers(0, N, size=N)
            bs_result = self._rebuild_result(result, idx)
            try:
                score = metric_fn(bs_result)
            except Exception:
                score = float('nan')
            bs_scores.append(score)
        arr = np.array(bs_scores)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
        }

    def _bootstrap_durability(self, r_D, r_Dprime, n=None):
        """Bootstrap DS with synchronized resampling of D and D' to preserve pairing."""
        n = n or self.config['bootstrap_n']
        N = r_D['N']
        rng = np.random.default_rng(43)
        bs_scores = []
        for _ in range(n):
            idx = rng.integers(0, N, size=N)
            bs_D = self._rebuild_result(r_D, idx)
            bs_Dp = self._rebuild_result(r_Dprime, idx)
            try:
                score = self.compute_durability_score(bs_D, bs_Dp)
            except Exception:
                score = float('nan')
            bs_scores.append(score)
        arr = np.array(bs_scores)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
        }

    def _bootstrap_mrs(self, r_D, r_Dprime, r_Dtilde, weights, n=None):
        """Bootstrap MRS with synchronized resampling across all three result dicts."""
        n = n or self.config['bootstrap_n']
        N = r_D['N']
        rng = np.random.default_rng(44)
        bs_scores = []
        for _ in range(n):
            idx = rng.integers(0, N, size=N)
            bs_D = self._rebuild_result(r_D, idx)
            bs_Dp = self._rebuild_result(r_Dprime, idx)
            bs_Dt = self._rebuild_result(r_Dtilde, idx)
            try:
                ps = self.compute_probability_score(bs_D)
                ds = self.compute_durability_score(bs_D, bs_Dp)
                deps = self.compute_dependability_score(bs_D)
                qots_d = self.compute_qots(bs_D)
                a_s = self.compute_availability_score(bs_Dt)
                mrs = self.compute_mrs([ps, ds, deps, qots_d['score'], a_s], weights)
            except Exception:
                mrs = float('nan')
            bs_scores.append(mrs)
        arr = np.array(bs_scores)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
        }

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def run_statistical_tests(self, r_D, r_Dprime):
        """Mann-Whitney U comparing failure confidences baseline vs shifted."""
        result = {}

        fc_D = r_D['failure_confidences']
        fc_Dp = r_Dprime['failure_confidences']

        if len(fc_D) >= 1 and len(fc_Dp) >= 1:
            try:
                stat, pval = mannwhitneyu(fc_D, fc_Dp, alternative='two-sided', use_continuity=True)
                result['mann_whitney_u'] = {
                    'description': 'Failure confidences: baseline D vs shifted D_prime',
                    'statistic': float(stat),
                    'pvalue': float(pval),
                    'significant_at_0.05': bool(pval < 0.05),
                }
            except Exception as exc:
                result['mann_whitney_u'] = {'error': str(exc)}
        else:
            result['mann_whitney_u'] = {
                'statistic': None, 'pvalue': None,
                'note': 'Insufficient failure samples for Mann-Whitney U test.',
            }

        return result

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def compute_all_metrics(self, loader_D, loader_Dprime, loader_Dtilde):
        """Run inference on all three datasets and compute the full MRS framework.

        Returns a comprehensive results dictionary suitable for JSON serialization.
        """
        print('  Running inference on baseline D ...')
        r_D = self._run_inference(loader_D)

        print('  Running inference on shifted D_prime ...')
        r_Dprime = self._run_inference(loader_Dprime)

        print('  Running inference on perturbed D_tilde ...')
        r_Dtilde = self._run_inference(loader_Dtilde)

        # --- Point estimates ---
        ps = self.compute_probability_score(r_D)
        ds = self.compute_durability_score(r_D, r_Dprime)
        deps = self.compute_dependability_score(r_D)
        qots_dict = self.compute_qots(r_D)
        qots = qots_dict['score']
        a_s = self.compute_availability_score(r_Dtilde)

        # DepS* intermediates for reporting
        c_bar = r_D['c_bar']
        rel_failure_confs = r_D['failure_confidences'] - c_bar if r_D['K'] > 0 else np.array([])
        var_95_cal = float(np.percentile(rel_failure_confs, 95)) if r_D['K'] > 0 else 0.0
        cvar_calibrated = float(self._compute_cvar(rel_failure_confs)) if r_D['K'] > 0 else 0.0
        ece = self._compute_ece(r_D['all_confidences'], r_D['failure_mask'])

        scores = [ps, ds, deps, qots, a_s]
        mrs_equal = self.compute_mrs(scores, self.config['weights_equal'])

        # --- Bootstrap CIs ---
        print('  Computing bootstrap confidence intervals ...')
        ps_ci = self.bootstrap_confidence_interval(r_D, self.compute_probability_score)
        ds_ci = self._bootstrap_durability(r_D, r_Dprime)
        deps_ci = self.bootstrap_confidence_interval(r_D, self.compute_dependability_score)
        qots_ci = self.bootstrap_confidence_interval(r_D, lambda r: self.compute_qots(r)['score'])
        as_ci = self.bootstrap_confidence_interval(r_Dtilde, self.compute_availability_score)
        mrs_equal_ci = self._bootstrap_mrs(r_D, r_Dprime, r_Dtilde, self.config['weights_equal'])

        # --- Statistical tests ---
        stat_tests = self.run_statistical_tests(r_D, r_Dprime)

        results = {
            'inference_summary': {
                'baseline_D': {
                    'N': r_D['N'], 'K': r_D['K'],
                    'raw_error_rate': r_D['K'] / r_D['N'] if r_D['N'] > 0 else 0.0,
                    'pf': r_D['pf'],
                    'accuracy_pct': 100.0 * (1.0 - r_D['K'] / r_D['N']) if r_D['N'] > 0 else 0.0,
                },
                'shifted_Dprime': {
                    'N': r_Dprime['N'], 'K': r_Dprime['K'],
                    'raw_error_rate': r_Dprime['K'] / r_Dprime['N'] if r_Dprime['N'] > 0 else 0.0,
                    'pf': r_Dprime['pf'],
                },
                'perturbed_Dtilde': {
                    'N': r_Dtilde['N'], 'K': r_Dtilde['K'],
                    'raw_error_rate': r_Dtilde['K'] / r_Dtilde['N'] if r_Dtilde['N'] > 0 else 0.0,
                    'pf': r_Dtilde['pf'],
                },
            },
            'metrics': {
                'PS': {'score': ps, 'bootstrap': ps_ci},
                'DS': {'score': ds, 'bootstrap': ds_ci},
                'DepS': {
                    'score': deps,
                    'c_bar': c_bar,
                    'var_95_calibrated': var_95_cal,
                    'cvar_calibrated': cvar_calibrated,
                    'ece': ece,
                    'bootstrap': deps_ci,
                },
                'QoTS': {
                    'score': qots,
                    'rho': qots_dict['rho'],
                    't_stat': qots_dict['t_stat'],
                    'p_value': qots_dict['p_value'],
                    'significant_at_0.05': qots_dict['significant_at_0.05'],
                    'pattern': qots_dict['pattern'],
                    'bootstrap': qots_ci,
                },
                'AS': {
                    'score': a_s,
                    'mtbf': float(np.mean(np.diff(r_Dtilde['failure_indices'])[np.diff(r_Dtilde['failure_indices']) > 1]))
                        if r_Dtilde['K'] >= 2 and any(np.diff(r_Dtilde['failure_indices']) > 1) else float(r_Dtilde['N']),
                    'mttr': float(np.mean(r_Dtilde['consecutive_failure_runs']))
                        if r_Dtilde['consecutive_failure_runs'] else 0.0,
                    'bootstrap': as_ci,
                },
            },
            'mrs': {
                'equal_weights': {
                    'weights': self.config['weights_equal'],
                    'score': mrs_equal,
                    'bootstrap': mrs_equal_ci,
                },
            },
            'statistical_tests': stat_tests,
            'config': {k: v for k, v in self.config.items()},
        }

        self._results = results
        return results

    # ------------------------------------------------------------------
    # Report printer
    # ------------------------------------------------------------------

    def print_report(self, results, model_name='Model'):
        """Print a formatted reliability report to stdout."""
        inf = results['inference_summary']
        m = results['metrics']
        mrs = results['mrs']
        st = results['statistical_tests']

        def ci_str(bs):
            if bs.get('ci_lower') is None or np.isnan(bs.get('ci_lower', float('nan'))):
                return 'N/A'
            return f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]"

        print()
        print('=' * 65)
        print(f'  MODEL RELIABILITY SCORE REPORT — {model_name}')
        print('=' * 65)

        # Inference summary
        d = inf['baseline_D']
        dp = inf['shifted_Dprime']
        dt = inf['perturbed_Dtilde']
        acc = d['accuracy_pct']
        print(f'\n  Dataset: CIFAR-100 (N={d["N"]:,} test samples)')
        print(f'  Accuracy (baseline D)   : {acc:.2f}%')
        print(f'  P(F|D)  (baseline)      : {d["pf"]:.4f}  (K={d["K"]:,} failures)')
        print(f'  P(F|D\') (shifted)       : {dp["pf"]:.4f}  (K={dp["K"]:,} failures)')
        print(f'  P(F|D~) (perturbed)     : {dt["pf"]:.4f}  (K={dt["K"]:,} failures)')

        # Sub-metrics
        print(f'\n  {"Metric":<10} {"Score":>8}  {"95% CI":>22}')
        print(f'  {"-"*44}')
        print(f'  {"P_s":<10} {m["PS"]["score"]:>8.4f}  {ci_str(m["PS"]["bootstrap"]):>22}')
        print(f'  {"DS":<10} {m["DS"]["score"]:>8.4f}  {ci_str(m["DS"]["bootstrap"]):>22}')
        print(f'  {"DepS*":<10} {m["DepS"]["score"]:>8.4f}  {ci_str(m["DepS"]["bootstrap"]):>22}')
        print(f'  {"QoTS":<10} {m["QoTS"]["score"]:>8.4f}  {ci_str(m["QoTS"]["bootstrap"]):>22}')
        print(f'  {"AS":<10} {m["AS"]["score"]:>8.4f}  {ci_str(m["AS"]["bootstrap"]):>22}')

        # DepS* diagnostics
        deps_m = m['DepS']
        print(f'\n  DepS* diagnostics:')
        print(f'    c_bar (global mean confidence)    : {deps_m["c_bar"]:.4f}')
        print(f'    VaR_0.95  (calibration-relative)  : {deps_m["var_95_calibrated"]:+.4f}')
        print(f'    CVaR_0.95 (calibration-relative)  : {deps_m["cvar_calibrated"]:+.4f}')
        print(f'    ECE (15 bins)                     : {deps_m["ece"]:.4f}')

        # QoTS diagnostics
        qots_m = m['QoTS']
        print(f'\n  QoTS diagnostics (lag-1 autocorrelation):')
        if qots_m['rho'] is not None:
            sig_str = 'significant' if qots_m['significant_at_0.05'] else 'not significant'
            print(f'    rho={qots_m["rho"]:+.4f}  t={qots_m["t_stat"]:.2f}  p={qots_m["p_value"]:.4f}  ({sig_str})')
            print(f'    Pattern: {qots_m["pattern"]}')
        else:
            print(f'    {qots_m["pattern"]}')
        print(f'    Note: Weibull analysis replaced — prior A-D test failed '
              f'(stat=273.74 >> crit=0.755, insufficient fit).')

        # MRS
        print()
        mq_eq = mrs['equal_weights']
        print(f'  MRS (equal weights) : {mq_eq["score"]:.4f}  {ci_str(mq_eq["bootstrap"])}')

        # Derived comparisons
        delta = m['PS']['score'] - m['AS']['score']
        print(f'\n  Accuracy vs P_s : {acc:.2f}% vs {m["PS"]["score"]:.4f}  '
              f'(P_s = 1 - P(F|D), accuracy = 1 - raw error rate)')
        print(f'  Delta = P_s - AS : {delta:+.4f}  '
              f'(positive = P_s > AS, robustness degrades under noise)')

        # Statistical tests
        mwu = st.get('mann_whitney_u', {})
        if mwu.get('pvalue') is not None:
            sig = 'significant' if mwu['significant_at_0.05'] else 'not significant'
            print(f'\n  Mann-Whitney U   : stat={mwu["statistic"]:.1f}  p={mwu["pvalue"]:.4f}  ({sig} at alpha=0.05)')
        else:
            print(f'\n  Mann-Whitney U   : {mwu.get("note", "N/A")}')

        print('=' * 65)
        print()
