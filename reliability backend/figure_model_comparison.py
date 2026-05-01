"""
figure_model_comparison.py — Model comparison: grouped bar chart + radar chart.

Panel A: grouped bars for all 5 sub-metrics + both MRS variants, with 95% CI error bars.
Panel B: radar chart for the 5 sub-metrics.

Outputs: figures/model_comparison_metrics.png
"""

import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('matplotlib is required: pip install matplotlib')

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_COLORS = {
    'AlexNet':  '#1f77b4',
    'VGG16':    '#ff7f0e',
    'ResNet50': '#2ca02c',
}
MODEL_ORDER = ['AlexNet', 'VGG16', 'ResNet50']

# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_results():
    path = os.path.join(RESULTS_DIR, 'results_overview.json')
    with open(path) as f:
        data = json.load(f)
    return {m['model']: m for m in data}


def _score_ci(entry):
    s  = entry['score']
    lo = entry.get('ci_lower')
    hi = entry.get('ci_upper')
    if lo is None: lo = s
    if hi is None: hi = s
    return s, max(s - lo, 0.0), max(hi - s, 0.0)


# -----------------------------------------------------------------------
# Panel A — Grouped bar chart
# -----------------------------------------------------------------------

# (display_label, section, key)
BAR_METRICS = [
    ('P_s',          'sub_metrics', 'P_s'),
    ('DS',           'sub_metrics', 'DS'),
    ('DepS*',        'sub_metrics', 'DepS_star'),
    ('QoTS',         'sub_metrics', 'QoTS'),
    ('AS',           'sub_metrics', 'AS'),
    ('MRS\n(equal)', 'MRS',         'equal_weights'),
    ('MRS\n(aero)',  'MRS',         'aerospace_weights'),
]


def plot_bars(ax, results):
    n_groups = len(BAR_METRICS)
    n_models = len(MODEL_ORDER)
    bar_w = 0.22
    x = np.arange(n_groups)

    for mi, name in enumerate(MODEL_ORDER):
        color  = MODEL_COLORS[name]
        model  = results[name]
        scores, errs_lo, errs_hi = [], [], []

        for _, section, key in BAR_METRICS:
            s, el, eh = _score_ci(model[section][key])
            scores.append(s)
            errs_lo.append(el)
            errs_hi.append(eh)

        offset = (mi - (n_models - 1) / 2.0) * bar_w
        ax.bar(x + offset, scores, bar_w, label=name, color=color,
               yerr=[errs_lo, errs_hi], capsize=3,
               error_kw={'linewidth': 0.9, 'ecolor': 'black', 'alpha': 0.7})

    ax.axhline(0.5, color='grey', linewidth=1.0, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _ in BAR_METRICS], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Sub-metric and MRS Scores with 95% CI', fontsize=11, pad=8)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# -----------------------------------------------------------------------
# Panel B — Radar chart
# -----------------------------------------------------------------------

RADAR_LABELS = ['P_s', 'DS', 'DepS*', 'QoTS', 'AS']
_RADAR_KEYS  = ['P_s', 'DS', 'DepS_star', 'QoTS', 'AS']


def plot_radar(ax, results):
    n = len(RADAR_LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed = angles + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for name in MODEL_ORDER:
        color = MODEL_COLORS[name]
        sub   = results[name]['sub_metrics']
        vals  = [sub[k]['score'] or 0.0 for k in _RADAR_KEYS]
        vals_c = vals + [vals[0]]

        ax.plot(closed, vals_c, color=color, linewidth=2.0, label=name)
        ax.fill(closed, vals_c, color=color, alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels(RADAR_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, color='grey')
    ax.grid(True, alpha=0.3)
    ax.set_title('Sub-metric Radar', fontsize=11, pad=18)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.35, 1.15))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    results = load_results()

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, polar=True)

    plot_bars(ax1, results)
    plot_radar(ax2, results)

    fig.tight_layout(pad=2.0)

    out = os.path.join(FIGURES_DIR, 'model_comparison_metrics.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {os.path.basename(out)}  (300 dpi)')
