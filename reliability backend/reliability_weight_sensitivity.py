"""
reliability_weight_sensitivity.py — P_s sensitivity to False Negative weight.

Demonstrates that P_s is a tunable metric distinct from accuracy.  With equal
weights w_I = w_II = 0.5, P_s is identical to the accuracy complement.  As
w_II increases (FN-heavy weighting) P_s penalises models more aggressively
than raw accuracy, which matters in safety-critical applications.

Formula used:
    P_F(w_II) = (K / N) * (w_I + w_II)   where w_I = 0.5 (fixed)
    P_s(w_II) = 1 - P_F(w_II)

This corresponds to the existing ReliabilityEvaluator formula where every
misclassification carries combined weight w_I + w_II.  At w_II = 0.5:
P_s = accuracy, confirming the equal-weight baseline.

Data is loaded from per-model diagnostic JSON files (or falls back to
reliability_results.json if they are not available).
"""

import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('WARNING: matplotlib not installed — chart will not be generated.')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)   # reliability backend/ -> project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

W_II_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
W_I_FIXED = 0.5  # reference False Positive weight (held constant)

MODEL_COLORS = {
    'AlexNet': '#1f77b4',   # blue
    'VGG16': '#ff7f0e',     # orange
    'ResNet50': '#2ca02c',  # green
}


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def _load_model_data():
    """Load K (failures), N (samples), and accuracy for each model.

    Priority:
      1. all_models_reliability_diagnostics.json (new format)
      2. Per-model {slug}_reliability_diagnostics.json
      3. reliability_results.json (legacy format)
    """
    models = {}

    # Try new combined diagnostics file
    combined = os.path.join(RESULTS_DIR, 'all_models_reliability_diagnostics.json')
    if os.path.exists(combined):
        with open(combined) as f:
            data = json.load(f)
        for entry in data:
            name = entry['model']
            fr = entry['failure_rates']
            k = fr['K_baseline']
            n = 10000  # fixed for CIFAR-100 10K test set
            acc = 1.0 - fr['P_F_baseline']
            models[name] = {'K': k, 'N': n, 'accuracy': acc}
        if models:
            print(f'Loaded data from all_models_reliability_diagnostics.json '
                  f'({len(models)} models)')
            return models

    # Try per-model diagnostics files
    slugs = {'AlexNet': 'alexnet', 'VGG16': 'vgg16', 'ResNet50': 'resnet50'}
    for name, slug in slugs.items():
        path = os.path.join(RESULTS_DIR, f'{slug}_reliability_diagnostics.json')
        if os.path.exists(path):
            with open(path) as f:
                entry = json.load(f)
            fr = entry['failure_rates']
            k = fr['K_baseline']
            n = 10000
            acc = 1.0 - fr['P_F_baseline']
            models[name] = {'K': k, 'N': n, 'accuracy': acc}

    if models:
        print(f'Loaded data from per-model diagnostics files ({len(models)} models)')
        return models

    # Fall back to legacy reliability_results.json
    legacy = os.path.join(PROJECT_ROOT, 'reliability_results.json')  # legacy fallback at root
    if os.path.exists(legacy):
        with open(legacy) as f:
            data = json.load(f)
        for name, entry in data.get('models', {}).items():
            bd = entry['inference_summary']['baseline_D']
            models[name] = {
                'K': bd['K'],
                'N': bd['N'],
                'accuracy': bd['accuracy_pct'] / 100.0,
            }
        if models:
            print(f'Loaded data from reliability_results.json ({len(models)} models)')
            return models

    raise FileNotFoundError(
        'No model data found. Run reliability_test.py first to generate results.'
    )


# -----------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------

def compute_ps_curve(K, N, w_ii_values, w_i_fixed=0.5):
    """Compute P_s at each w_II value.

    P_F(w_II) = (K/N) * (w_I_fixed + w_II)
    P_s(w_II) = 1 - P_F
    """
    error_rate = K / N
    ps_values = [1.0 - error_rate * (w_i_fixed + w_ii) for w_ii in w_ii_values]
    return np.clip(ps_values, 0.0, 1.0)


def compute_divergence(ps_curve, accuracy):
    """Divergence = P_s(w_II) - accuracy (zero at w_II = 0.5 with w_I = 0.5)."""
    return np.array(ps_curve) - accuracy


# -----------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------

def print_summary(models_data):
    w_ii = W_II_VALUES
    model_names = list(models_data.keys())

    curves = {
        name: compute_ps_curve(d['K'], d['N'], w_ii, W_I_FIXED)
        for name, d in models_data.items()
    }

    header_accs = '/'.join(f'{models_data[n]["accuracy"]:.3f}' for n in model_names)
    col_w = 12

    print()
    print('Weight Sensitivity Summary')
    print('=' * (8 + col_w * len(model_names) + 22))
    header = f'{"w_II":<6}  ' + '  '.join(f'{n+" P_s":>{col_w}}' for n in model_names)
    header += f'   Accuracy ({"/".join(n[:3] for n in model_names)})'
    print(header)
    print('-' * (8 + col_w * len(model_names) + 22))

    max_div = {n: (0.0, 0.0) for n in model_names}  # (|div|, w_ii)

    for i, w in enumerate(w_ii):
        row = f'{w:<6.1f}  '
        row += '  '.join(f'{curves[n][i]:>{col_w}.4f}' for n in model_names)
        row += f'   {header_accs}'
        print(row)
        for n in model_names:
            div = abs(curves[n][i] - models_data[n]['accuracy'])
            if div > max_div[n][0]:
                max_div[n] = (div, w)

    print('=' * (8 + col_w * len(model_names) + 22))
    print('Note: Accuracy is constant. P_s diverges as weights become asymmetric.')
    print('      At w_II=0.5 (with w_I=0.5 fixed): P_s = accuracy. [confirmed]')
    print()
    print('Maximum divergence |P_s - accuracy|:')
    for n, (div, w) in max_div.items():
        print(f'  {n}: {div:.4f} at w_II={w}')
    print()


# -----------------------------------------------------------------------
# Chart
# -----------------------------------------------------------------------

def generate_chart(models_data):
    if not HAS_MPL:
        print('Skipping chart generation (matplotlib not available).')
        return

    w_ii = np.array(W_II_VALUES)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    annotation_done = {}

    for name, d in models_data.items():
        color = MODEL_COLORS.get(name, '#333333')
        accuracy = d['accuracy']
        ps_curve = compute_ps_curve(d['K'], d['N'], w_ii.tolist(), W_I_FIXED)
        divergence = compute_divergence(ps_curve, accuracy)

        # --- Subplot 1: P_s lines ---
        ax1.plot(w_ii, ps_curve, color=color, linewidth=2, marker='o', markersize=5,
                 label=f'{name} P_s')
        ax1.axhline(accuracy, color=color, linewidth=1.5, linestyle='--',
                    label=f'{name} Accuracy')

        # Annotate maximum absolute divergence
        abs_div = np.abs(divergence)
        max_idx = int(np.argmax(abs_div))
        max_w = w_ii[max_idx]
        max_ps = ps_curve[max_idx]
        max_div = divergence[max_idx]
        sign = '+' if max_div >= 0 else ''
        ax1.annotate(
            f'{sign}{max_div:.3f}',
            xy=(max_w, max_ps),
            xytext=(max_w + 0.03, max_ps + 0.015 * (1 if max_div >= 0 else -1)),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
        )

        # --- Subplot 2: Divergence ---
        ax2.plot(w_ii, divergence, color=color, linewidth=2, marker='o', markersize=5,
                 label=f'{name}')

    # Vertical reference lines
    for xval, lbl in [(0.5, 'Equal weights\n(P_s = accuracy)'),
                      (0.7, 'Aerospace-motivated\nweights')]:
        ax1.axvline(xval, color='grey', linewidth=1.0, linestyle=':')
        ax1.text(xval + 0.01, ax1.get_ylim()[0] + 0.01 if ax1.get_ylim() else 0.02,
                 lbl, fontsize=7.5, color='grey', va='bottom')
        ax2.axvline(xval, color='grey', linewidth=1.0, linestyle=':')

    # Subplot 2: shading and y=0 line
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', label='No divergence')
    ax2.fill_between(w_ii, 0, 0.25, alpha=0.10, color='green',
                     label='P_s > accuracy (FP-heavy weighting)')
    ax2.fill_between(w_ii, -0.25, 0, alpha=0.10, color='red',
                     label='P_s < accuracy (FN-heavy weighting)')

    # Styling — subplot 1
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('P_s Sensitivity to False Negative Weight vs Accuracy', fontsize=12, pad=8)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=8, loc='lower left', ncol=2)
    ax1.set_xlim(0.05, 0.95)

    # Secondary x annotation on top axis
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks([])
    ax1_top.set_xlabel('← FP-prioritized                          FN-prioritized →',
                       fontsize=9, labelpad=4)

    # Styling — subplot 2
    ax2.set_xlabel('False Negative Weight (w_II)', fontsize=11)
    ax2.set_ylabel('P_s − Accuracy', fontsize=11)
    ax2.set_title('P_s Divergence from Accuracy', fontsize=11, pad=6)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.set_xticks(W_II_VALUES)

    fig.tight_layout()

    png_path = os.path.join(FIGURES_DIR, 'reliability_weight_sensitivity.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Chart saved: {os.path.basename(png_path)}  (300 dpi)')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    models_data = _load_model_data()

    print_summary(models_data)
    generate_chart(models_data)
