"""
Model Reliability Score (MRS) framework.

Reframes misclassification as a failure event and applies reliability
engineering principles across five components:
  P_s   — Probability Score    (baseline failure rate)
  DS    — Durability Score      (degradation under distribution shift)
  DepS* — Dependability Score   (calibration-relative CVaR of failure confidence)
  QoTS  — Quality over Time     (threshold-gated Weibull of critical failure intervals)
  AS    — Availability Score    (1 - P(F|D_tilde), perturbed failure rate)
  MRS   — weighted harmonic mean of all five (four when QoTS is None)
"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, weibull_min, gamma as gamma_dist, lognorm, anderson


_DEFAULT_CONFIG = {
    'w_I': 0.5,
    'w_II': 0.5,
    'alpha': 0.95,
    'ece_bins': 15,
    'sigma': 0.25,
    'weights_equal': [0.2, 0.2, 0.2, 0.2, 0.2],
    'bootstrap_n': 1000,
    'qots_theta': 0.90,
    'qots_min_failures': 10,
    'lambda_param': 2.0,
    'gamma_param': 1.0,
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

        Every misclassification is simultaneously a Type I FP for the predicted
        class and a Type II FN for the true class.
        w(tau_n) = w_I + w_II — with equal defaults (0.5 + 0.5 = 1.0) this yields
        P(F|D) = K/N, matching raw accuracy (P_s = 1 - error_rate).
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
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                confs = probs.max(dim=1).values

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_confidences.extend(confs.cpu().tolist())

        all_preds = np.array(all_preds, dtype=np.int64)
        all_labels = np.array(all_labels, dtype=np.int64)
        all_confidences = np.array(all_confidences, dtype=np.float64)

        failure_mask = all_preds != all_labels
        failure_indices = np.where(failure_mask)[0]
        failure_confidences = all_confidences[failure_mask]

        N = len(all_preds)
        K = int(failure_mask.sum())
        pf = (K / N) * w_avg if N > 0 else 0.0

        return {
            'N': N,
            'K': K,
            'pf': pf,
            'c_bar': float(np.mean(all_confidences)),
            'failure_mask': failure_mask,
            'all_confidences': all_confidences,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'failure_indices': failure_indices,
            'failure_confidences': failure_confidences,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_cvar(self, confidences):
        """CVaR at config['alpha'] on an array of (possibly calibration-relative) scores."""
        alpha = self.config['alpha']
        if len(confidences) == 0:
            return 0.0
        sorted_c = np.sort(confidences)
        n = len(sorted_c)
        cutoff_idx = min(int(np.floor(alpha * n)), n - 1)
        return float(np.mean(sorted_c[cutoff_idx:]))

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
            if i == n_bins - 1:
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
        return {
            'N': N, 'K': K, 'pf': pf,
            'c_bar': float(np.mean(ac)),
            'failure_mask': fm,
            'all_confidences': ac,
            'failure_indices': fi,
            'failure_confidences': fc,
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
            return 1.0
        ds = 1.0 - (pf_Dp - pf_D) / pf_D
        return float(np.clip(ds, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 3 — Dependability Score
    # ------------------------------------------------------------------

    def compute_dependability_score(self, r_D):
        """DepS* = calibration-adjusted dependability using relative failure confidence.

        Uses calibration-relative confidence c_n* = c_n - c_bar at each failure,
        where c_bar is the global mean confidence over all N inferences.

        DepS* = 1 - (clip(CVaR_alpha(F*), -1, 1) / 2 + 0.5)
          CVaR = +1  (failures far above average confidence) -> DepS* = 0.0
          CVaR =  0  (failures at average confidence)        -> DepS* = 0.5
          CVaR = -1  (failures below average confidence)     -> DepS* = 1.0
        """
        if r_D['K'] == 0:
            return 1.0
        c_bar = r_D['c_bar']
        rel_confs = r_D['failure_confidences'] - c_bar
        cvar = self._compute_cvar(rel_confs)
        deps = 1.0 - (float(np.clip(cvar, -1.0, 1.0)) / 2.0 + 0.5)
        return float(np.clip(deps, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 4 — Quality over Time Score (threshold-gated Weibull)
    # ------------------------------------------------------------------

    def compute_qots(self, r_D):
        """QoTS via threshold-gated Weibull analysis of critical failure intervals.

        Critical failures: T = {n | failure AND c_n > theta, theta = qots_theta}.
        Overconfident misclassifications are rare enough for Weibull's
        rare-event assumption to hold.

        Fit Weibull, Gamma, and Lognormal to inter-critical-failure intervals
        (floc=0, MLE); select best model by AIC.

        QoTS = best_dist.sf(mu_ref)
          mu_ref = gamma_param * N / K_all  (expected interval between all failures)
        High QoTS: critical failures are well-spaced relative to the baseline failure rate.

        Returns {'score': None, ...} when |T| < qots_min_failures.
        """
        theta = self.config['qots_theta']
        min_failures = self.config['qots_min_failures']
        gamma_p = self.config['gamma_param']

        failure_mask = r_D['failure_mask']
        all_confs = r_D['all_confidences']
        N = r_D['N']
        K_all = r_D['K']

        critical_indices = np.where(failure_mask & (all_confs > theta))[0]
        K_crit = len(critical_indices)

        base = {
            'score': None,
            'K_critical': K_crit,
            'theta': theta,
            'note': f'Insufficient critical failures: {K_crit} < {min_failures} (theta={theta})',
        }

        if K_crit < min_failures or len(critical_indices) < 2:
            return base

        intervals = np.diff(critical_indices).astype(np.float64)
        if len(intervals) == 0:
            return base

        # --- Fit three distributions (floc=0 → 2 free parameters each) ---
        aics = {}
        wb_params = gm_params = ln_params = None

        try:
            c, loc, scale = weibull_min.fit(intervals, floc=0)
            ll = float(np.sum(weibull_min.logpdf(intervals, c, loc=loc, scale=scale)))
            aics['weibull'] = 2 * 2 - 2 * ll
            wb_params = (c, loc, scale)
        except Exception:
            aics['weibull'] = np.inf

        try:
            a, loc, scale = gamma_dist.fit(intervals, floc=0)
            ll = float(np.sum(gamma_dist.logpdf(intervals, a, loc=loc, scale=scale)))
            aics['gamma'] = 2 * 2 - 2 * ll
            gm_params = (a, loc, scale)
        except Exception:
            aics['gamma'] = np.inf

        try:
            s, loc, scale = lognorm.fit(intervals, floc=0)
            ll = float(np.sum(lognorm.logpdf(intervals, s, loc=loc, scale=scale)))
            aics['lognormal'] = 2 * 2 - 2 * ll
            ln_params = (s, loc, scale)
        except Exception:
            aics['lognormal'] = np.inf

        best_model = min(aics, key=aics.get)
        if aics[best_model] == np.inf:
            return {**base, 'note': 'All distribution fits failed.'}

        # --- Anderson-Darling goodness-of-fit ---
        ad_stat = ad_crit = ad_sig = None
        try:
            ad_result = anderson(intervals, dist='weibull_min')
            ad_stat = float(ad_result.statistic)
            ad_crit = [float(c) for c in ad_result.critical_values]
            ad_sig = [float(s) for s in ad_result.significance_level]
        except (ValueError, AttributeError):
            try:
                log_ivs = np.log(np.clip(intervals, 1e-10, None))
                ad_result = anderson(log_ivs, dist='gumbel_r')
                ad_stat = float(ad_result.statistic)
                ad_crit = [float(c) for c in ad_result.critical_values]
                ad_sig = [float(s) for s in ad_result.significance_level]
            except Exception:
                pass

        # --- QoTS score: survival at reference interval ---
        mu_ref = gamma_p * float(N / K_all) if K_all > 0 else float(N)

        try:
            if best_model == 'weibull' and wb_params is not None:
                qots = float(weibull_min.sf(mu_ref, *wb_params))
            elif best_model == 'gamma' and gm_params is not None:
                qots = float(gamma_dist.sf(mu_ref, *gm_params))
            elif ln_params is not None:
                qots = float(lognorm.sf(mu_ref, *ln_params))
            else:
                qots = 0.5
        except Exception:
            qots = 0.5
        qots = float(np.clip(qots, 0.0, 1.0))

        return {
            'score': qots,
            'K_critical': K_crit,
            'theta': theta,
            'intervals_n': len(intervals),
            'intervals_mean': float(np.mean(intervals)),
            'mu_ref': mu_ref,
            'best_model': best_model,
            'aic': aics,
            'weibull_shape': float(wb_params[0]) if wb_params else None,
            'weibull_scale': float(wb_params[2]) if wb_params else None,
            'ad_stat': ad_stat,
            'ad_crit_vals': ad_crit,
            'ad_sig_levels': ad_sig,
            'reference_shape': self.config['lambda_param'],
        }

    # ------------------------------------------------------------------
    # Metric 5 — Availability Score
    # ------------------------------------------------------------------

    def compute_availability_score(self, r_Dtilde):
        """AS = 1 - P(F|D_tilde).

        Measures availability under stochastic degradation.  Identical formula
        to P_s but evaluated on the perturbed dataset D_tilde.
        """
        return float(np.clip(1.0 - r_Dtilde['pf'], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Composite MRS
    # ------------------------------------------------------------------

    def compute_mrs(self, scores, weights):
        """Weighted harmonic mean over non-None scores.

        None scores are excluded and the remaining weights are used as-is;
        the harmonic mean formula (sum_w / sum_w/s) is proportional so it
        naturally renormalizes without explicit weight rescaling.
        """
        pairs = [(s, w) for s, w in zip(scores, weights) if s is not None]
        w = np.array([w for _, w in pairs], dtype=np.float64)
        s = np.array([max(sc, 1e-9) for sc, _ in pairs], dtype=np.float64)
        return float(np.sum(w) / np.sum(w / s))

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_confidence_interval(self, result, metric_fn, n=None):
        """Resample N samples with replacement and recompute metric_fn 1000 times.

        metric_fn must accept a result dict (not a loader) so inference is not
        repeated on each bootstrap iteration.  None return values are treated
        as NaN and filtered before computing CI statistics.
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
            bs_scores.append(float('nan') if score is None else score)
        arr = np.array(bs_scores)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'),
                    'ci_lower': float('nan'), 'ci_upper': float('nan')}
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
            return {'mean': float('nan'), 'std': float('nan'),
                    'ci_lower': float('nan'), 'ci_upper': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
        }

    def _bootstrap_mrs(self, r_D, r_Dprime, r_Dtilde, weights, n=None):
        """Bootstrap MRS with synchronized resampling across all three result dicts.

        Handles QoTS=None: those bootstrap iterations use 4-metric harmonic mean.
        """
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
            return {'mean': float('nan'), 'std': float('nan'),
                    'ci_lower': float('nan'), 'ci_upper': float('nan')}
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

    def compute_all_metrics(self, loader_D, loader_Dprime, loader_Dtilde, dataset_info=None):
        """Run inference on all three datasets and compute the full MRS framework.

        Args:
            loader_D, loader_Dprime, loader_Dtilde: DataLoaders for the three variants.
            dataset_info: Optional dict from get_reliability_loaders (transform_dist, dist_stats).

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
        n_active = sum(1 for s in scores if s is not None)

        # --- Bootstrap CIs ---
        print('  Computing bootstrap confidence intervals ...')
        ps_ci = self.bootstrap_confidence_interval(r_D, self.compute_probability_score)
        ds_ci = self._bootstrap_durability(r_D, r_Dprime)
        deps_ci = self.bootstrap_confidence_interval(r_D, self.compute_dependability_score)
        qots_ci = (
            self.bootstrap_confidence_interval(r_D, lambda r: self.compute_qots(r)['score'])
            if qots is not None else None
        )
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
                    'accuracy_pct': 100.0 * (1.0 - r_Dtilde['K'] / r_Dtilde['N']) if r_Dtilde['N'] > 0 else 0.0,
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
                    'K_critical': qots_dict.get('K_critical'),
                    'theta': qots_dict.get('theta'),
                    'best_model': qots_dict.get('best_model'),
                    'aic': qots_dict.get('aic'),
                    'weibull_shape': qots_dict.get('weibull_shape'),
                    'weibull_scale': qots_dict.get('weibull_scale'),
                    'intervals_mean': qots_dict.get('intervals_mean'),
                    'mu_ref': qots_dict.get('mu_ref'),
                    'ad_stat': qots_dict.get('ad_stat'),
                    'ad_crit_vals': qots_dict.get('ad_crit_vals'),
                    'note': qots_dict.get('note'),
                    'bootstrap': qots_ci,
                },
                'AS': {
                    'score': a_s,
                    'pf_dtilde': r_Dtilde['pf'],
                    'K_dtilde': r_Dtilde['K'],
                    'bootstrap': as_ci,
                },
            },
            'mrs': {
                'equal_weights': {
                    'weights': self.config['weights_equal'],
                    'active_metrics': n_active,
                    'score': mrs_equal,
                    'bootstrap': mrs_equal_ci,
                },
            },
            'statistical_tests': stat_tests,
            'config': {k: v for k, v in self.config.items()},
        }

        if dataset_info is not None:
            results['dataset_info'] = {
                'transform_dist': dataset_info.get('transform_dist', {}),
                'dist_stats': dataset_info.get('dist_stats', {}),
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
            if bs is None:
                return 'N/A'
            if bs.get('ci_lower') is None or (
                    isinstance(bs.get('ci_lower'), float) and bs['ci_lower'] != bs['ci_lower']):
                return 'N/A'
            return f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]"

        print()
        print('=' * 65)
        print(f'  MODEL RELIABILITY SCORE REPORT — {model_name}')
        print('=' * 65)

        d = inf['baseline_D']
        dp = inf['shifted_Dprime']
        dt = inf['perturbed_Dtilde']
        acc = d['accuracy_pct']
        acc_dt = dt.get('accuracy_pct', 0.0)

        print(f'\n  Dataset: CIFAR-100 (N={d["N"]:,} test samples)')
        print(f'  Accuracy (baseline D)       : {acc:.2f}%')
        print(f'  Accuracy (perturbed D_tilde): {acc_dt:.2f}%')
        print(f'  P(F|D)  (baseline)          : {d["pf"]:.4f}  (K={d["K"]:,} failures)')
        print(f'  P(F|D\') (FFT shifted)        : {dp["pf"]:.4f}  (K={dp["K"]:,} failures)')
        print(f'  P(F|D~) (stoch degraded)    : {dt["pf"]:.4f}  (K={dt["K"]:,} failures)')

        # Sub-metrics
        qots_score_str = f'{m["QoTS"]["score"]:>8.4f}' if m["QoTS"]["score"] is not None else '     N/A'
        print(f'\n  {"Metric":<10} {"Score":>8}  {"95% CI":>22}')
        print(f'  {"-"*44}')
        print(f'  {"P_s":<10} {m["PS"]["score"]:>8.4f}  {ci_str(m["PS"]["bootstrap"]):>22}')
        print(f'  {"DS":<10} {m["DS"]["score"]:>8.4f}  {ci_str(m["DS"]["bootstrap"]):>22}')
        print(f'  {"DepS*":<10} {m["DepS"]["score"]:>8.4f}  {ci_str(m["DepS"]["bootstrap"]):>22}')
        print(f'  {"QoTS":<10} {qots_score_str}  {ci_str(m["QoTS"]["bootstrap"]):>22}')
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
        print(f'\n  QoTS diagnostics (threshold-gated Weibull):')
        print(f'    Critical failure threshold theta  : {qots_m["theta"]}')
        print(f'    K_critical (overconfident misclass): {qots_m["K_critical"]}')
        if qots_m['score'] is None:
            print(f'    {qots_m.get("note", "Insufficient data")}')
        else:
            print(f'    Best-fit distribution             : {qots_m["best_model"]}')
            if qots_m.get('aic'):
                for mdl, aic_val in qots_m['aic'].items():
                    marker = ' <-- best' if mdl == qots_m['best_model'] else ''
                    print(f'    AIC [{mdl:<10}]               : {aic_val:.1f}{marker}')
            if qots_m.get('weibull_shape') is not None:
                print(f'    Weibull shape (c)                 : {qots_m["weibull_shape"]:.4f}')
                print(f'    Weibull scale (lambda)            : {qots_m["weibull_scale"]:.2f}')
            if qots_m.get('intervals_mean') is not None:
                print(f'    Mean inter-critical interval      : {qots_m["intervals_mean"]:.1f}')
                print(f'    Reference interval mu_ref         : {qots_m["mu_ref"]:.1f}')
            if qots_m.get('ad_stat') is not None:
                print(f'    A-D statistic                     : {qots_m["ad_stat"]:.4f}')

        # AS
        as_m = m['AS']
        print(f'\n  AS diagnostics (availability under degradation):')
        print(f'    P(F|D_tilde)                      : {as_m["pf_dtilde"]:.4f}  (K={as_m["K_dtilde"]:,})')
        print(f'    AS = 1 - P(F|D_tilde)             : {as_m["score"]:.4f}')

        # MRS
        n_active = mrs['equal_weights'].get('active_metrics', 5)
        note = '' if n_active == 5 else f'  (renormalized over {n_active} metrics — QoTS excluded)'
        print()
        mq_eq = mrs['equal_weights']
        print(f'  MRS (equal weights) : {mq_eq["score"]:.4f}  {ci_str(mq_eq["bootstrap"])}{note}')

        # Summary comparison
        delta = m['PS']['score'] - m['AS']['score']
        print(f'\n  Accuracy vs P_s : {acc:.2f}% vs {m["PS"]["score"]:.4f}')
        print(f'  Delta P_s - AS  : {delta:+.4f}  '
              f'(positive = reliability drops under stochastic degradation)')

        # Dataset info (if available)
        if 'dataset_info' in results:
            di = results['dataset_info']
            if di.get('dist_stats'):
                ds_d = di['dist_stats'].get('D', {})
                ds_dp = di['dist_stats'].get('D_prime', {})
                print(f'\n  Spectral shift (D vs D_prime FFT):')
                print(f'    HF energy ratio D       : {ds_d.get("hf_energy_ratio", 0):.6f}')
                print(f'    HF energy ratio D_prime : {ds_dp.get("hf_energy_ratio", 0):.6f}')

        # Statistical tests
        mwu = st.get('mann_whitney_u', {})
        if mwu.get('pvalue') is not None:
            sig = 'significant' if mwu['significant_at_0.05'] else 'not significant'
            print(f'\n  Mann-Whitney U   : stat={mwu["statistic"]:.1f}  '
                  f'p={mwu["pvalue"]:.4f}  ({sig} at alpha=0.05)')
        else:
            print(f'\n  Mann-Whitney U   : {mwu.get("note", "N/A")}')

        print('=' * 65)
        print()
