"""
reliability_test.py — Entry point for the MRS evaluation framework.

Loads each trained model, runs the full reliability evaluation against three
dataset variants, prints a clean report, and saves two JSON files per model:
  {model}_reliability_scores.json      — scores and CIs only
  {model}_reliability_diagnostics.json — intermediate values and diagnostics

After all models, writes combined:
  all_models_reliability_scores.json
  all_models_reliability_diagnostics.json
"""

import os
import sys
import json
import warnings
import importlib.util
from datetime import datetime

import torch

warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchvision')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained models')
RELIABILITY_DIR = os.path.join(PROJECT_ROOT, 'reliability backend')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, RELIABILITY_DIR)

from reliability_datasets import get_reliability_loaders
from reliability_metrics import ReliabilityEvaluator

REGISTRY = [
    {
        'name': 'AlexNet',
        'file': os.path.join(MODELS_DIR, 'alexnet.py'),
        'class': 'AlexNet',
        'checkpoint': os.path.join(MODELS_DIR, 'alexnet.pth'),
    },
    {
        'name': 'VGG16',
        'file': os.path.join(MODELS_DIR, 'vggnet.py'),
        'class': 'VGG16',
        'checkpoint': os.path.join(MODELS_DIR, 'vgg16.pth'),
    },
    {
        'name': 'ResNet50',
        'file': os.path.join(MODELS_DIR, 'resnet.py'),
        'class': 'ResNet50',
        'checkpoint': os.path.join(MODELS_DIR, 'resnet50.pth'),
    },
]


def load_model_class(name, file_path, class_name):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _make_json_safe(obj):
    """Recursively convert numpy/torch scalars to Python native types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, float) and (obj != obj):
        return None
    return obj


def _ci(bs):
    """Extract (ci_lower, ci_upper) from a bootstrap dict, converting NaN to None."""
    if bs is None:
        return None, None
    lo = bs.get('ci_lower')
    hi = bs.get('ci_upper')
    if isinstance(lo, float) and lo != lo:
        lo = None
    if isinstance(hi, float) and hi != hi:
        hi = None
    return lo, hi


def _build_scores_dict(model_name, results):
    """Build the compact scores dict matching reliability_scores.json schema."""
    m = results['metrics']
    mrs = results['mrs']
    inf = results['inference_summary']

    ps_lo, ps_hi = _ci(m['PS']['bootstrap'])
    ds_lo, ds_hi = _ci(m['DS']['bootstrap'])
    deps_lo, deps_hi = _ci(m['DepS']['bootstrap'])
    qots_lo, qots_hi = _ci(m['QoTS'].get('bootstrap'))
    as_lo, as_hi = _ci(m['AS']['bootstrap'])
    eq_lo, eq_hi = _ci(mrs['equal_weights']['bootstrap'])
    ae_lo, ae_hi = _ci(mrs['aerospace_weights']['bootstrap'])

    return {
        'model': model_name,
        'dataset': 'CIFAR-100',
        'n_samples': inf['baseline_D']['N'],
        'accuracy': inf['baseline_D']['accuracy_pct'] / 100.0,
        'sub_metrics': {
            'P_s': {
                'score': m['PS']['score'],
                'ci_lower': ps_lo,
                'ci_upper': ps_hi,
            },
            'DS': {
                'score': m['DS']['score'],
                'ci_lower': ds_lo,
                'ci_upper': ds_hi,
            },
            'DepS_star': {
                'score': m['DepS']['score'],
                'ci_lower': deps_lo,
                'ci_upper': deps_hi,
            },
            'QoTS': {
                'score': m['QoTS']['score'],
                'ci_lower': qots_lo,
                'ci_upper': qots_hi,
            },
            'AS': {
                'score': m['AS']['score'],
                'ci_lower': as_lo,
                'ci_upper': as_hi,
            },
        },
        'MRS': {
            'equal_weights': {
                'score': mrs['equal_weights']['score'],
                'ci_lower': eq_lo,
                'ci_upper': eq_hi,
                'weights': mrs['equal_weights']['weights'],
            },
            'aerospace_weights': {
                'score': mrs['aerospace_weights']['score'],
                'ci_lower': ae_lo,
                'ci_upper': ae_hi,
                'weights': mrs['aerospace_weights']['weights'],
            },
        },
    }


def _build_diagnostics_dict(model_name, results, dataset_info):
    """Build the full diagnostics dict matching reliability_diagnostics.json schema."""
    inf = results['inference_summary']
    m = results['metrics']
    st = results['statistical_tests']
    cfg = results['config']
    qots_m = m['QoTS']
    aic = qots_m.get('aic') or {}

    pf_D = inf['baseline_D']['pf']
    pf_Dp = inf['shifted_Dprime']['pf']
    deg_ratio = (pf_Dp - pf_D) / pf_D if pf_D > 0 else 0.0

    trans_dist_raw = (dataset_info or {}).get('transform_dist', {})
    n_total = sum(trans_dist_raw.values()) if trans_dist_raw else 1
    trans_dist = {
        k: {'count': v, 'pct': round(100.0 * v / n_total, 1)}
        for k, v in trans_dist_raw.items()
    }

    dist_stats = (dataset_info or {}).get('dist_stats', {})
    hf_baseline = dist_stats.get('D', {}).get('hf_energy_ratio')
    hf_shifted = dist_stats.get('D_prime', {}).get('hf_energy_ratio')

    ps_score = m['PS']['score']
    as_score = m['AS']['score']
    delta = ps_score - as_score
    if delta > 0.005:
        delta_interp = 'P_s > AS: reliability degrades under stochastic degradation'
    elif delta < -0.005:
        delta_interp = 'P_s < AS: model is more robust on degraded data than baseline'
    else:
        delta_interp = 'No meaningful difference between baseline and degraded reliability'

    mwu = st.get('mann_whitney_u', {})

    return {
        'model': model_name,
        'dataset': 'CIFAR-100',
        'failure_rates': {
            'P_F_baseline': pf_D,
            'P_F_shifted': pf_Dp,
            'P_F_perturbed': inf['perturbed_Dtilde']['pf'],
            'K_baseline': inf['baseline_D']['K'],
            'K_shifted': inf['shifted_Dprime']['K'],
            'K_perturbed': inf['perturbed_Dtilde']['K'],
        },
        'dataset_conditions': {
            'baseline': 'Clean CIFAR-100 test set',
            'durability': 'FFT low-frequency reconstruction radius=8',
            'availability': 'Stochastic pixel-space degradation',
            'spectral_shift': {
                'HF_energy_baseline': hf_baseline,
                'HF_energy_shifted': hf_shifted,
            },
            'transform_distribution': trans_dist,
        },
        'P_s_diagnostics': {
            'w_I': cfg['w_I'],
            'w_II': cfg['w_II'],
        },
        'DS_diagnostics': {
            'degradation_ratio': deg_ratio,
        },
        'DepS_diagnostics': {
            'c_bar': m['DepS']['c_bar'],
            'VaR_0_95': m['DepS']['var_95_calibrated'],
            'CVaR_0_95': m['DepS']['cvar_calibrated'],
            'ECE': m['DepS']['ece'],
        },
        'QoTS_diagnostics': {
            'theta': qots_m.get('theta'),
            'K_critical': qots_m.get('K_critical'),
            'best_fit_distribution': qots_m.get('best_model'),
            'AIC_weibull': aic.get('weibull'),
            'AIC_gamma': aic.get('gamma'),
            'AIC_lognormal': aic.get('lognormal'),
            'beta_w': qots_m.get('weibull_shape'),
            'eta': qots_m.get('weibull_scale'),
            'mean_inter_critical_interval': qots_m.get('intervals_mean'),
            'AD_statistic': qots_m.get('ad_stat'),
            'failure_mechanism': qots_m.get('failure_mechanism'),
            'beta_w_note': qots_m.get('beta_w_note'),
        },
        'AS_diagnostics': {
            'P_F_D_tilde': m['AS']['pf_dtilde'],
            'delta_Ps_AS': float(delta),
            'delta_interpretation': delta_interp,
        },
        'statistical_tests': {
            'mann_whitney_stat': mwu.get('statistic'),
            'mann_whitney_p': mwu.get('pvalue'),
            'mann_whitney_significant': mwu.get('significant_at_0.05'),
        },
        'bootstrap': {
            'n_resamples': cfg['bootstrap_n'],
            'confidence_level': 0.95,
        },
    }


def _write_json(path, data):
    with open(path, 'w') as f:
        json.dump(_make_json_safe(data), f, indent=2)
    print(f'  Saved: {os.path.basename(path)}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\nPreparing reliability dataloaders ...')
    loader_D, loader_Dprime, loader_Dtilde, dataset_info = get_reliability_loaders(
        data_dir=os.path.join(PROJECT_ROOT, 'data'),
        batch_size=128,
        num_workers=0,
    )
    print(f'  Baseline D       : {len(loader_D.dataset):,} images')
    print(f'  Shifted D_prime  : {len(loader_Dprime.dataset):,} images')
    print(f'  Perturbed D_tilde: {len(loader_Dtilde.dataset):,} images')
    print(f'\n  D_tilde transform distribution:')
    for tname, count in sorted(dataset_info['transform_dist'].items()):
        print(f'    {tname:<25}: {count:,}')

    all_scores = []
    all_diagnostics = []

    timestamp = datetime.now().isoformat(timespec='seconds')

    for cfg in REGISTRY:
        print(f'\n{"="*67}')
        print(f'  Evaluating: {cfg["name"]}')
        print(f'{"="*67}')

        if not os.path.exists(cfg['checkpoint']):
            print(f'  WARNING: Checkpoint not found — skipping {cfg["name"]}')
            continue

        ModelClass = load_model_class(cfg['name'], cfg['file'], cfg['class'])
        model = ModelClass(num_classes=100).to(device)
        model.load_state_dict(
            torch.load(cfg['checkpoint'], map_location=device, weights_only=True)
        )
        print(f'  Loaded checkpoint: {os.path.basename(cfg["checkpoint"])}')

        evaluator = ReliabilityEvaluator(model, device, config=None)
        results = evaluator.compute_all_metrics(
            loader_D, loader_Dprime, loader_Dtilde, dataset_info=dataset_info
        )
        evaluator.print_report(results, model_name=cfg['name'])

        slug = cfg['name'].lower().replace(' ', '_')
        scores_dict = _build_scores_dict(cfg['name'], results)
        diag_dict = _build_diagnostics_dict(cfg['name'], results, dataset_info)

        print(f'\n  Writing output files for {cfg["name"]} ...')
        _write_json(os.path.join(PROJECT_ROOT, f'{slug}_reliability_scores.json'), scores_dict)
        _write_json(os.path.join(PROJECT_ROOT, f'{slug}_reliability_diagnostics.json'), diag_dict)

        all_scores.append(scores_dict)
        all_diagnostics.append(diag_dict)

    # ------------------------------------------------------------------
    # Combined files
    # ------------------------------------------------------------------
    if all_scores:
        print('\nWriting combined output files ...')
        _write_json(os.path.join(PROJECT_ROOT, 'all_models_reliability_scores.json'), all_scores)
        _write_json(os.path.join(PROJECT_ROOT, 'all_models_reliability_diagnostics.json'), all_diagnostics)
    else:
        print('\nNo models were evaluated. Train at least one model and re-run.')
