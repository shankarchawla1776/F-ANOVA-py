import numpy as np
import matplotlib.pyplot as plt


def plot_test_stats(p_value, alpha, null_dist, test_stat, test_name, hypothesis, hypothesis_label):

    if p_value <= alpha:
        line_label = f'{test_name} Statistic P-Value: p={p_value} <= {alpha}'
        verdict_label = 'Verdict: Reject H_0'

    else:
        line_label = f'{test_name} Statistic P-Value: p={p_value} > {alpha}'
        verdict_label = 'Verdict: Fail to Reject H_0'

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f'Null Distribution Plot for {test_name}')

    n, bins, patches = ax.hist(null_dist, bins=100, density=True, alpha=0.7, label='Null Distribution', color='grey', edgecolor='black')

    ax.axvline(test_stat, color='red', linestyle='--', linewidth=1.5, label=line_label)

    if (test_stat > np.max(null_dist) * 1e2) or (test_stat < np.min(null_dist) * 1e2):
        ax.set_xscale('log')
        ax.minorticks_on()

    crit_value = np.quantile(null_dist, 1 - alpha)
    ylim_max = ax.get_ylim()[1]
    xlim_max = ax.get_xlim()[1]

    x_fill = np.linspace(crit_value, xlim_max, 100)
    y_fill = np.ones_like(x_fill) * ylim_max
    ax.fill_between(x_fill, 0, ylim_max, alpha=0.3, color='red', label='Region for Statistical Significance (Reject H₀)')

    ax.axvline(crit_value, color='black', linestyle='-', linewidth=1.5, label='Beginning of Critical Value Region')

    if 'f' in test_name.lower():
        null_dist_label = 'Simulated F-type Mixture Null Distribution'
        if hypothesis.upper() == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: F-type test'
        elif hypothesis.upper() == "PAIRWISE":
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: F-type test'
    else:
        null_dist_label = 'Simulated X^2_{1}-type Mixture Null Distribution'
        if hypothesis.upper() == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: Squared L-2 Norm test'
        elif hypothesis.upper() == "PAIRWISE":
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: Squared L-2 Norm test'

    ax.set_ylabel('PDF', fontsize=10)
    ax.set_xlabel('Null Distribution', fontsize=10)
    ax.set_title(f'{title_label}, {verdict_label}', fontsize=10)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)

    ax.tick_params(labelsize=18)

    plt.tight_layout()

    plt.show()

    return fig, ax
