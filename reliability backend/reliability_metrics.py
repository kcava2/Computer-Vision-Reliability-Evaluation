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
    'weights_aerospace': [0.15, 0.15, 0.25, 0.25, 0.20],
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
        """Expected Calibration Error over equal-width confidence bins."""
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
        """P_s = 1 - P(F|D)."""
        return float(np.clip(1.0 - r_D['pf'], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 2 — Durability Score
    # ------------------------------------------------------------------

    def compute_durability_score(self, r_D, r_Dprime):
        """DS = clip(1 - (P(F|D') - P(F|D)) / P(F|D), 0, 1)."""
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
        """DepS* = 1 - (clip(CVaR_alpha(F*), -1, 1) / 2 + 0.5)

        F* = failure_confidences - c_bar  (calibration-relative).
        """
        if r_D['K'] == 0:
            return 1.0
        c_bar = r_D['c_bar']
        rel_confs = r_D['failure_confidences'] - c_bar
        cvar = self._compute_cvar(rel_confs)
        deps = 1.0 - (float(np.clip(cvar, -1.0, 1.0)) / 2.0 + 0.5)
        return float(np.clip(deps, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Metric 4 — Quality over Time Score (threshold-gated Weibull, Option B)
    # ------------------------------------------------------------------

    def compute_qots(self, r_D):
        """QoTS via threshold-gated Weibull/Gamma/Lognormal AIC selection.

        Critical failures: T = {n | failure AND c_n > theta}.
        Fit Weibull, Gamma, and Lognormal to inter-critical-failure intervals
        (floc=0, MLE); select best by AIC.

        Option B shape-based scoring:
          Weibull/Gamma: QoTS = exp(-lambda * max(shape-1, 0)) *
                                exp(-gamma  * max(1-shape, 0))
          Lognormal:     direction from mu_ln vs log(mean_interval);
                         wear-out  -> exp(-lambda * sigma)
                         infant    -> exp(-gamma  * sigma)

        Weibull beta_w is always reported as a reference diagnostic.
        Returns {'score': None, ...} when |T| < qots_min_failures.
        """
        theta = self.config['qots_theta']
        min_failures = self.config['qots_min_failures']
        lambda_p = self.config['lambda_param']
        gamma_p = self.config['gamma_param']

        failure_mask = r_D['failure_mask']
        all_confs = r_D['all_confidences']
        N = r_D['N']

        critical_indices = np.where(failure_mask & (all_confs > theta))[0]
        K_crit = len(critical_indices)

        base = {
            'score': None,
            'K_critical': K_crit,
            'theta': theta,
            'weibull_shape': None,
            'weibull_scale': None,
            'failure_mechanism': None,
            'beta_w_note': None,
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
            return {**base, 'aic': aics, 'note': 'All distribution fits failed.'}

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

        # --- Option B: shape-based scoring ---
        mean_iv = float(np.mean(intervals))
        qots = 0.5
        failure_mechanism = 'Unknown'
        beta_w_note = None

        if best_model == 'weibull' and wb_params is not None:
            shape = float(wb_params[0])
            qots = (np.exp(-lambda_p * max(shape - 1.0, 0.0)) *
                    np.exp(-gamma_p * max(1.0 - shape, 0.0)))
            if shape < 0.9:
                diag = f'infant mortality / decreasing hazard (beta_w={shape:.3f})'
            elif shape > 1.1:
                diag = f'wear-out / increasing hazard (beta_w={shape:.3f})'
            else:
                diag = f'approximately constant hazard (beta_w={shape:.3f})'
            failure_mechanism = f'Failure mechanism (weibull): {diag}'
            beta_w_note = 'Weibull is the best-fit distribution.'

        elif best_model == 'gamma' and gm_params is not None:
            shape = float(gm_params[0])
            qots = (np.exp(-lambda_p * max(shape - 1.0, 0.0)) *
                    np.exp(-gamma_p * max(1.0 - shape, 0.0)))
            if shape < 0.9:
                diag = f'decreasing hazard (alpha={shape:.3f})'
            elif shape > 1.1:
                diag = f'increasing hazard (alpha={shape:.3f})'
            else:
                diag = f'approximately constant hazard (alpha={shape:.3f})'
            failure_mechanism = f'Failure mechanism (gamma): {diag}'
            beta_w_note = ('Weibull shape parameter reported as reference. '
                           'QoTS computed using best-fit distribution (gamma).')

        elif best_model == 'lognormal' and ln_params is not None:
            sigma = float(ln_params[0])
            scale_ln = float(ln_params[2])
            mu_ln = float(np.log(max(scale_ln, 1e-10)))
            log_mean_iv = float(np.log(max(mean_iv, 1e-10)))
            if mu_ln > log_mean_iv:
                pattern = 'wear-out analog'
                qots = np.exp(-lambda_p * sigma)
            else:
                pattern = 'infant mortality analog'
                qots = np.exp(-gamma_p * sigma)
            failure_mechanism = (f'Failure mechanism (lognormal): {pattern} '
                                 f'(mu_ln={mu_ln:.3f}, log(mean_iv)={log_mean_iv:.3f}, '
                                 f'sigma={sigma:.3f})')
            beta_w_note = ('Weibull shape parameter reported as reference. '
                           'QoTS computed using best-fit distribution (lognormal).')

        qots = float(np.clip(qots, 1e-9, 1.0))

        beta_w_ref = float(wb_params[0]) if wb_params else None
        eta_ref = float(wb_params[2]) if wb_params else None

        return {
            'score': qots,
            'K_critical': K_crit,
            'theta': theta,
            'intervals_n': len(intervals),
            'intervals_mean': mean_iv,
            'best_model': best_model,
            'aic': aics,
            'weibull_shape': beta_w_ref,
            'weibull_scale': eta_ref,
            'beta_w_note': beta_w_note,
            'failure_mechanism': failure_mechanism,
            'ad_stat': ad_stat,
            'ad_crit_vals': ad_crit,
            'ad_sig_levels': ad_sig,
            'note': None,
        }

    # ------------------------------------------------------------------
    # Metric 5 — Availability Score
    # ------------------------------------------------------------------

    def compute_availability_score(self, r_Dtilde):
        """AS = 1 - P(F|D_tilde)."""
        return float(np.clip(1.0 - r_Dtilde['pf'], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Composite MRS
    # ------------------------------------------------------------------

    _METRIC_NAMES = ['P_s', 'DS', 'DepS*', 'QoTS', 'AS']

    def compute_mrs(self, scores, weights):
        """Weighted harmonic mean. None scores are excluded; remaining weights renormalize
        automatically via the sum(w)/sum(w/s) formula."""
        pairs = []
        for i, (s, w) in enumerate(zip(scores, weights)):
            if s is None:
                continue
            if s == 0.0:
                name = self._METRIC_NAMES[i] if i < len(self._METRIC_NAMES) else f'metric_{i}'
                print(f'WARNING: {name} = 0.0 exactly. Using epsilon=1e-6 for MRS computation. '
                      f'Investigate metric computation.')
                s = 1e-6
            pairs.append((s, w))
        w = np.array([w for _, w in pairs], dtype=np.float64)
        s = np.array([sc for sc, _ in pairs], dtype=np.float64)
        return float(np.sum(w) / np.sum(w / s))

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_confidence_interval(self, result, metric_fn, n=None):
        """Resample N samples with replacement and recompute metric_fn 1000 times."""
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

    def _bootstrap_qots(self, r_D, n=None):
        """Bootstrap QoTS with per-iteration retry when a resample produces fewer than
        10 critical-failure intervals.  Retries up to 100 times before skipping the
        iteration.  Reports the fraction of successful resamples."""
        n = n or self.config['bootstrap_n']
        N = r_D['N']
        rng = np.random.default_rng(42)
        min_intervals = 10
        max_retries = 100
        bs_scores = []
        n_skipped = 0

        for _ in range(n):
            score = None
            for attempt in range(max_retries + 1):
                idx = rng.integers(0, N, size=N)
                bs_r = self._rebuild_result(r_D, idx)
                qots_d = self.compute_qots(bs_r)
                if qots_d['score'] is not None:
                    intervals_n = qots_d.get('intervals_n', 0)
                    if intervals_n is None or intervals_n < min_intervals:
                        continue
                    score = float(qots_d['score'])
                    break
            if score is None:
                n_skipped += 1
            else:
                bs_scores.append(score)

        if n_skipped > 0:
            frac = n_skipped / n
            print(f'  QoTS bootstrap: {n_skipped}/{n} iterations skipped '
                  f'({frac:.1%} insufficient intervals after {max_retries} retries).')

        arr = np.array(bs_scores) if bs_scores else np.array([])
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
        """Bootstrap DS with synchronized resampling to preserve pairing."""
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

    def _bootstrap_mrs(self, r_D, r_Dprime, r_Dtilde, weights, n=None, seed=44):
        """Bootstrap MRS with synchronized resampling across all three result dicts."""
        n = n or self.config['bootstrap_n']
        N = r_D['N']
        rng = np.random.default_rng(seed)
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
        """Run inference on all three datasets and compute the full MRS framework."""
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

        # DepS* intermediates
        c_bar = r_D['c_bar']
        rel_failure_confs = r_D['failure_confidences'] - c_bar if r_D['K'] > 0 else np.array([])
        var_95_cal = float(np.percentile(rel_failure_confs, 95)) if r_D['K'] > 0 else 0.0
        cvar_calibrated = float(self._compute_cvar(rel_failure_confs)) if r_D['K'] > 0 else 0.0
        ece = self._compute_ece(r_D['all_confidences'], r_D['failure_mask'])

        scores = [ps, ds, deps, qots, a_s]
        n_active = sum(1 for s in scores if s is not None)
        mrs_equal = self.compute_mrs(scores, self.config['weights_equal'])
        mrs_aero = self.compute_mrs(scores, self.config['weights_aerospace'])

        # --- Bootstrap CIs ---
        print('  Computing bootstrap confidence intervals ...')
        ps_ci = self.bootstrap_confidence_interval(r_D, self.compute_probability_score)
        ds_ci = self._bootstrap_durability(r_D, r_Dprime)
        deps_ci = self.bootstrap_confidence_interval(r_D, self.compute_dependability_score)
        qots_ci = self._bootstrap_qots(r_D) if qots is not None else None
        as_ci = self.bootstrap_confidence_interval(r_Dtilde, self.compute_availability_score)
        mrs_equal_ci = self._bootstrap_mrs(r_D, r_Dprime, r_Dtilde, self.config['weights_equal'],
                                           seed=44)
        mrs_aero_ci = self._bootstrap_mrs(r_D, r_Dprime, r_Dtilde, self.config['weights_aerospace'],
                                          seed=45)

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
                    'failure_mechanism': qots_dict.get('failure_mechanism'),
                    'beta_w_note': qots_dict.get('beta_w_note'),
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
                'aerospace_weights': {
                    'weights': self.config['weights_aerospace'],
                    'active_metrics': n_active,
                    'score': mrs_aero,
                    'bootstrap': mrs_aero_ci,
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
    # Report printer — clean scores-only format
    # ------------------------------------------------------------------

    def print_report(self, results, model_name='Model'):
        """Print a clean reliability report showing only scores and CIs."""
        m = results['metrics']
        mrs = results['mrs']
        inf = results['inference_summary']
        acc = inf['baseline_D']['accuracy_pct'] / 100.0
        n = inf['baseline_D']['N']
        qots_score = m['QoTS']['score']
        qots_none = qots_score is None
        n_active = mrs['equal_weights'].get('active_metrics', 5)

        def _ci(bs):
            if bs is None:
                return float('nan'), float('nan')
            lo = bs.get('ci_lower', float('nan'))
            hi = bs.get('ci_upper', float('nan'))
            if lo != lo: lo = float('nan')
            if hi != hi: hi = float('nan')
            return lo, hi

        def score_row(label, score, bs):
            if score is None:
                return f'  {label:<32} {"N/A":>8}   [insufficient critical failures]'
            lo, hi = _ci(bs)
            if lo != lo:
                ci_s = '[N/A]'
            else:
                ci_s = f'[{lo:.4f}, {hi:.4f}]'
            return f'  {label:<32} {score:>8.4f}   {ci_s}'

        def mrs_row(label, score, bs, tag=''):
            lo, hi = _ci(bs)
            if lo != lo:
                ci_s = '[N/A]'
            else:
                ci_s = f'[{lo:.4f}, {hi:.4f}]'
            return f'  {label:<32} {score:>8.4f}   {ci_s}{tag}'

        W = 67
        tag = '  [4-metric]' if n_active < 5 else ''

        print()
        print('=' * W)
        print(f'  MODEL RELIABILITY SCORE REPORT — {model_name}')
        print('=' * W)
        print(f'  Dataset : CIFAR-100  (N={n:,})')
        print(f'  Accuracy: {acc:.2%}')
        print()
        print(f'  {"Sub-Metric":<32} {"Score":>8}   95% CI')
        print(f'  {"-"*(W-2)}')
        print(score_row('Probability Score (P_s)', m['PS']['score'], m['PS']['bootstrap']))
        print(score_row('Durability Score (DS)', m['DS']['score'], m['DS']['bootstrap']))
        print(score_row('Dependability Score (DepS*)', m['DepS']['score'], m['DepS']['bootstrap']))
        print(score_row('Quality Score (QoTS)', qots_score, m['QoTS'].get('bootstrap')))
        print(score_row('Availability Score (AS)', m['AS']['score'], m['AS']['bootstrap']))
        print()
        print(f'  {"-"*(W-2)}')
        print(mrs_row('MRS (equal weights)', mrs['equal_weights']['score'],
                      mrs['equal_weights']['bootstrap'], tag))
        print(mrs_row('MRS (aerospace weights)', mrs['aerospace_weights']['score'],
                      mrs['aerospace_weights']['bootstrap'], tag))
        print('=' * W)
        print()
