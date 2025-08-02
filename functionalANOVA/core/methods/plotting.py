import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import math
from functionalANOVA.core import utils


# TODO: Needs work
def plot_means(self, plot_type='default', subgroup_indicator=None, observation_size_label=True, group_labels=None, primary_labels=None, secondary_labels=None, x_scale='', y_scale='', domain_units_label='', response_units_label='', data_transparency=0.1, legend_transparency=0.3333, data_line_width=1.75, mean_line_width=5, font_size=18, title_labels=None, save_path='', legend_location='best', num_columns=1, legend_title='', new_colors=None, position=(90, 90, 1400, 800)):

    def _legend_locations():
        return [
            'upper right',
            'upper left',
            'lower left',
            'lower right',
            'right',
            'center left',
            'center right',
            'lower center',
            'upper center',
            'center',
            'best'
        ]

    if subgroup_indicator is not None:
        self._groups.subgroup_indicator = subgroup_indicator
    if group_labels is not None:
        self._labels.group = group_labels
    if primary_labels is not None:
        self._labels.primary = primary_labels
    if secondary_labels is not None:
        self._labels.secondary = secondary_labels
    if domain_units_label:
        self._units.domain = domain_units_label
    if response_units_label:
        self._units.response = response_units_label

    plot_type = plot_type.upper()

    domain_label = self._labels.domain or ''
    response_label = self._labels.response or ''

    if not self._groups.subgroup_indicator:

        assert plot_type == 'DEFAULT', 'TwoWay plotting options require a subgroup_indicator argument'
        the_labels = self._labels.group
        n_labels = self.n_i
    else:
        set_up_two_way(self)

        if plot_type in ['DEFAULT', 'PRIMARY']:
            plot_type = 'PRIMARY'
            the_labels = self._labels.primary
            n_labels = self.n_i

        elif plot_type == "SECONDARY":
            the_labels = self._labels.secondary
            n_labels = np.zeros(self._groups.B)
            for k in range(self._groups.A):
                for kk in range(self._groups.B):
                    n_labels[kk] += self.n_ii[k][kk]

        elif plot_type == "INTERACTION":
            the_labels = generate_two_way_comb(self)
            n_labels = np.concatenate([item for sublist in self.n_ii for item in sublist])

    if observation_size_label:
        if self._labels.generic_group:
            the_data_labels = [f": Group Data ({int(n)})" for n in n_labels]
        else:
            the_data_labels = [f": Data ({int(n)})" for n in n_labels]
    else:
        the_data_labels = [''] * len(n_labels)

    if getattr(self, 'generic_group_labels', True):
        mean_group_labels = [f"{label}: Group Mean" for label in the_labels]
        data_group_labels = [f"{the_labels[i]}{the_data_labels[i]}" for i in range(len(the_labels))]
    else:
        mean_group_labels = [f"{label}: Mean" for label in the_labels]
        data_group_labels = [f"{the_labels[i]}{the_data_labels[i]}" for i in range(len(the_labels))]

    title_labels_str = ''
    save_labels = ''
    class_type = ''

    if hasattr(self, 'echo_ensemble_recs') and self.echo_ensemble_recs is not None:
        if title_labels:
            title_labels_str = self.echo_ensemble_recs.make_summary_string(title_labels, True)
            save_labels = self.echo_ensemble_recs.make_summary_string(title_labels, True, sanitize_string=True)
        class_type = type(self.echo_ensemble_recs.pull_sample.sources).__name__
    else:
        if title_labels:
            title_labels_str = str(title_labels)
            save_labels = str(title_labels)

    # fast patch for figure scale errors
    fig_width = max(position[2] / 100, 8)
    fig_height = max(position[3] / 100, 6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if plot_type == 'DEFAULT':
        mean_groups = [np.mean(data, axis=1) for data in self.data]

        if new_colors is None:
            if self._groups.k < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, self._groups.k))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, self._groups.k))
        else:
            assert new_colors.shape[1] == 3, 'Color order matrix must have 3 columns representing RGB values from 0 to 1'
            assert new_colors.shape[0] >= self._groups.k, f'Color order matrix must have at least {self._groups.k} rows representing all the groups'
            color_list = new_colors

        plot_reals = []
        legend_example_lines = []

        for k in range(self._groups.k):
            lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k], linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(self._groups.k):
            line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k],
                          linewidth=mean_line_width, linestyle='--')
            plot_means.append(line)

        ax.set_title(f'F-ANOVA Group Means and Realizations {title_labels_str}', fontsize=font_size)
        lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=legend_location, ncol=num_columns)

        if legend_title:
            lg.set_title(legend_title)

    elif plot_type == 'PRIMARY':
        mean_groups = [np.mean(data, axis=1) for data in self.data]

        if new_colors is None:
            if self._groups.k < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, self._groups.k))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, self._groups.k))
        else:
            color_list = new_colors

        plot_reals = []
        legend_example_lines = []

        for k in range(self._groups.k):
            lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k], linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(self._groups.k):
            line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k], linewidth=mean_line_width, linestyle='--')
            plot_means.append(line)

        ax.set_title(f'F-ANOVA Primary Factor Means and Realizations {title_labels_str}', fontsize=font_size)
        lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=legend_location, ncol=num_columns)

        if not legend_title:
            lg.set_title("Primary Factor Levels")
        else:
            lg.set_title(legend_title)

    elif plot_type == 'SECONDARY':
        yy = np.concatenate([data.T for data in self.data], axis=0)

        bflag = self._groups.subgroup_indicator
        bflag0 = np.unique(bflag)
        n_secondary = len(bflag0)

        if new_colors is None:
            if n_secondary < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, n_secondary))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, n_secondary))
        else:
            color_list = new_colors

        sub_data = []
        mean_groups = []

        for k in range(n_secondary):
            flag = bflag == bflag0[k]
            sub_data_k = yy[flag, :].T
            sub_data.append(sub_data_k)
            mean_groups.append(np.mean(sub_data_k, axis=1))

        plot_reals = []
        legend_example_lines = []

        for k in range(n_secondary):
            lines = ax.plot(self.d_grid, sub_data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k], linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(n_secondary):
            line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k],
                          linewidth=mean_line_width, linestyle='--')
            plot_means.append(line)

        ax.set_title(f'F-ANOVA Secondary Factor Means and Realizations {title_labels_str}', fontsize=font_size)
        lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=legend_location, ncol=num_columns)

        if not legend_title:
            lg.set_title("Secondary Factor Levels")
        else:
            lg.set_title(legend_title)

    elif plot_type == 'INTERACTION':
        yy = np.concatenate([data.T for data in self.data], axis=0)

        aflag = np.repeat(np.arange(1, self._groups.A + 1), self.n_i)
        bflag = self._groups.subgroup_indicator
        aflag0 = np.unique(aflag)
        bflag0 = np.unique(bflag)
        p, q = len(aflag0), len(bflag0)
        ab = p * q

        combinations = generate_two_way_comb(self)
        mean_labels = [f"{comb}: Mean" for comb in combinations]

        if observation_size_label:
            data_labels = [f"{combinations[k]}: Data ({int(n_labels[k])})" for k in range(len(combinations))]
        else:
            data_labels = [f"{comb}: Data" for comb in combinations]

        if new_colors is None:
            if ab < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, ab))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, ab))
        else:
            color_list = new_colors

        sub_data = []
        mean_groups = []
        counter = 0

        for i in aflag0:
            for j in bflag0:
                ijflag = (aflag == i) & (bflag == j)
                yyi = yy[ijflag, :]
                sub_data.append(yyi.T)
                mean_groups.append(np.mean(yyi, axis=0))
                counter += 1

        plot_reals = []
        legend_example_lines = []

        for k in range(ab):
            lines = ax.plot(self.d_grid, sub_data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k],linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(ab):
            line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k], linewidth=mean_line_width, linestyle='--')
            plot_means.append(line)

        ax.set_title(f'F-ANOVA Primary and Secondary Factor Combinatorial Means and Realizations {title_labels_str}', fontsize=font_size)
        lg = ax.legend(plot_means + legend_example_lines, mean_labels + data_labels, loc=legend_location, ncol=num_columns)

        if not legend_title:
            lg.set_title("TwoWay Factor Levels")
        else:
            lg.set_title(legend_title)

    if hasattr(self, 'echo_ensemble_recs') and self.echo_ensemble_recs is not None:
        if self._units.domain:
            ax.set_xlabel(f'{domain_label} ({self._units.domain})', fontsize=font_size)
        else:
            ax.set_xlabel(domain_label, fontsize=font_size)

        if self._units.response:
            ax.set_ylabel(f'{response_label} ({self._units.response})', fontsize=font_size)
        else:
            ax.set_ylabel(response_label, fontsize=font_size)
    else:
        if self._units.domain:
            ax.set_xlabel(f'({self._units.domain})', fontsize=font_size)

        if self._units.response:
            ax.set_ylabel(f'({self._units.response})', fontsize=font_size)

    if (
        class_type.lower() == 'psdrecord'
        or class_type.lower() == 'srsrecord'
        or class_type.lower() == 'spectralrecord'
        or (
            'srs' in response_label.lower()
            and (not x_scale or not y_scale)
        )
    ):
        if not x_scale:
            ax.set_xscale('log')
        else:
            ax.set_xscale(x_scale)

        if not y_scale:
            ax.set_yscale('log')
        else:
            ax.set_yscale(y_scale)
    else:
        if not x_scale:
            ax.set_xscale('linear')
        else:
            ax.set_xscale(x_scale)

        if not y_scale:
            ax.set_yscale('linear')
        else:
            ax.set_yscale(y_scale)

    ax.tick_params(labelsize=font_size)

    if save_path and os.path.isdir(save_path):

        save_filename = f"GroupMeans_{save_labels}.png"

        fig.savefig(os.path.join(save_path, save_filename))
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


# TODO: Needs work
def plot_covariances(self, plot_type='default', subgroup_indicator=None, group_labels=None, primary_labels=None, secondary_labels=None, x_scale='', y_scale='', color_scale='', domain_units_label='', response_units_label='', title_labels=None, save_path='', position=(90, 257, 2000, 800)):

    def _make_covariances(data, k, domain_points):
        eta_i = np.zeros((domain_points, k))
        v_hat_i = [None] * k

        gamma_hat_i = np.zeros((domain_points, domain_points, k))
        n_ii = np.zeros(k)

        for kk in range(k):
            n_i = data[kk].shape[1]
            n_ii[kk] = n_i
            eta_i[:, kk] = np.mean(data[kk], axis=1)
            zero_mean_data_k_subset = data[kk] - eta_i[:, kk:kk+1]
            v_hat_i[kk] = zero_mean_data_k_subset
            gamma_hat_i[:, :, kk] = (1 / (n_i - 1)) * (zero_mean_data_k_subset @ zero_mean_data_k_subset.T)

        N = int(np.sum(n_ii))
        pooled_covar_terms = np.zeros((domain_points, domain_points, k))

        for kk in range(k):
            pooled_covar_terms[:, :, kk] = (n_ii[kk] - 1) * gamma_hat_i[:, :, kk]
        pooled_covar = np.sum(pooled_covar_terms, axis=2) / (N - k)

        return gamma_hat_i, pooled_covar, n_ii

    def _grid_generator(k):
        if k <= 3:
            n = k
            m = 1
        elif k == 4:
            n = 2
            m = 2
        elif k == 5:
            n = 2
            m = 3
        elif k == 6:
            n = 2
            m = 3
        elif k == 7:
            n = 3
            m = 3
        elif k == 8:
            n = 3
            m = 3
        else:
            # n = math.ceil(math.sqrt(k))
            # m = math.ceil(k / n)
            n = np.ceil(math.sqrt(k))
            m = np.ceil(k / n)
        return m, n

    if subgroup_indicator is not None:
        self._groups.subgroup_indicator = subgroup_indicator
    if group_labels is not None:
        self._labels.group = group_labels
    if primary_labels is not None:
        self._labels.primary = primary_labels
    if secondary_labels is not None:
        self._labels.secondary = secondary_labels
    if domain_units_label:
        self._units.domain = domain_units_label
    if response_units_label:
        self._units.response = response_units_label

    plot_type = plot_type.upper()
    fig_label = 'Group'
    temp_label = ''

    if not self._groups.subgroup_indicator:
        assert plot_type == 'DEFAULT', 'TwoWay plotting options require a subgroup_indicator argument'

        if self._labels.generic_group:
            display_label = [f"Group {i+1}" for i in range(self._groups.k)]
        else:
            display_label = self._labels.group
    else:
        set_up_two_way(self)

        if plot_type in ['DEFAULT', 'PRIMARY']:
            plot_type = 'PRIMARY'
            fig_label = 'Primary Factor'
            display_label = self.primary_labels

            if self._labels.generic_group:
                display_label = [f"Group {label}" for label in display_label]

        elif plot_type == 'SECONDARY':
            fig_label = 'Secondary Factor'
            display_label = self.secondary_labels

            if self._labels.generic_group:
                display_label = [f"Group {label}" for label in display_label]

        elif plot_type == 'INTERACTION':
            fig_label = 'Primary & Secondary Factor'
            combinations = generate_two_way_comb(self)
            display_label = combinations

    if hasattr(self, 'echo_ensemble_recs') and self.echo_ensemble_recs is not None:
        if title_labels:
            title_labels_str = self.echo_ensemble_recs.make_summary_string(title_labels, True)
            save_labels = self.echo_ensemble_recs.make_summary_string(title_labels, True, sanitize_string=True)
        else:
            title_labels_str = ''
            save_labels = ''

    else:
        if title_labels:
            title_labels_str = str(title_labels)
            save_labels = str(title_labels)
        else:
            title_labels_str = ''
            save_labels = ''

    if not self._units.response:
        color_bar_label = '(Response)^2'
    else:
        color_bar_label = f'({self._units.response})^2'

    if plot_type in ['DEFAULT', 'PRIMARY']:
        gamma_hat_i, pooled_covar, n_ii = _make_covariances(self.data, self._groups.k, self.n_domain_points)
        K = self._groups.k

    elif plot_type == 'SECONDARY':
        yy = np.concatenate([data.T for data in self.data], axis=0)

        bflag = self._groups.subgroup_indicator
        bflag0 = np.unique(bflag)

        n_secondary = len(bflag0)
        sub_data = []

        for k in range(n_secondary):
            flag = bflag == bflag0[k]
            sub_data.append(yy[flag, :].T)

        gamma_hat_i, pooled_covar, n_ii = _make_covariances(sub_data, n_secondary, self.n_domain_points)
        K = n_secondary

    elif plot_type == 'INTERACTION':
        yy = np.concatenate([data.T for data in self.data], axis=0)
        aflag = np.repeat(np.arange(1, self._groups.A + 1), self.n_i)
        bflag = self._groups.subgroup_indicator

        aflag0 = np.unique(aflag)
        bflag0 = np.unique(bflag)

        p, q = len(aflag0), len(bflag0)
        ab = p * q
        self._groups.AB = ab
        sub_data = []
        counter = 0

        for i in aflag0:
            for j in bflag0:
                ijflag = (aflag == i) & (bflag == j)
                yyi = yy[ijflag, :]
                sub_data.append(yyi.T)
                counter += 1

        gamma_hat_i, pooled_covar, n_ii = _make_covariances(sub_data, ab, self.n_domain_points)
        K = ab

    fig_width = max(position[2] / 100, 8)  # minimum 8 inches
    fig_height = max(position[3] / 100, 6)  # minimum 6 inches

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.canvas.manager.set_window_title(f'{fig_label} Covariances Visualized')

    n_plots = K + 1
    rows, cols = _grid_generator(n_plots)

    for ii in range(K):
        ax = plt.subplot(rows, cols, ii + 1)
        cmin = np.min(gamma_hat_i[:, :, ii])
        cmax = np.max(gamma_hat_i[:, :, ii])

        im = ax.imshow(gamma_hat_i[:, :, ii], extent=[self.d_grid[0], self.d_grid[-1], self.d_grid[0], self.d_grid[-1]],
                      origin='lower', aspect='equal', vmin=cmin, vmax=cmax)
        ax.set_title(f"{display_label[ii]} | n = {int(n_ii[ii])}")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(ax.get_yticks())

        if hasattr(self, 'echo_ensemble_recs') and self.echo_ensemble_recs is not None:
            if not self._units.domain:
                temp_label = self._labels.domain or ''
            else:
                temp_label = f"{self._labels.domain or ''} ({self._units.domain})"
        else:
            if self._units.domain:
                temp_label = f"({self._units.domain})"

        ax.set_xlabel(temp_label)
        ax.set_ylabel(temp_label)

        if not x_scale:
            ax.set_xscale('linear')
        else:
            ax.set_xscale(x_scale)

        if not y_scale:
            ax.set_yscale('linear')
        else:
            ax.set_yscale(y_scale)

    ax = plt.subplot(rows, cols, K + 1)
    cmin = np.min(pooled_covar)
    cmax = np.max(pooled_covar)

    im = ax.imshow(pooled_covar, extent=[self.d_grid[0], self.d_grid[-1], self.d_grid[0], self.d_grid[-1]],
                  origin='lower', aspect='equal', vmin=cmin, vmax=cmax)

    ax.set_title(f"Pooled | n = {self.N}")
    ax.set_xlabel(temp_label)
    ax.set_ylabel(temp_label)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(ax.get_yticks())

    plt.suptitle(f'{fig_label} Covariances {title_labels_str}', fontsize=10)

    for ax in fig.get_axes():
        ax.tick_params(labelsize=18)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(18)

    plt.tight_layout()

    if not x_scale:
        plt.gca().set_xscale('linear')
    else:
        plt.gca().set_xscale(x_scale)

    if not y_scale:
        plt.gca().set_yscale('linear')
    else:
        plt.gca().set_yscale(y_scale)

    if not color_scale:
        pass

    else:
        if hasattr(plt.gca(), 'set_norm'):
            from matplotlib.colors import LogNorm
            if color_scale == 'log':
                plt.gca().images[0].set_norm(LogNorm())

    if save_path and os.path.isdir(save_path):
        save_filename = f"Covariance_{save_labels}.png"
        fig.savefig(os.path.join(save_path, save_filename))
        plt.close(fig)
    else:
        plt.show()

    return fig

# TODO: Some option or ability to save these plots
def plot_test_stats(p_value, alpha, null_dist, test_stat, test_name, hypothesis, hypothesis_label):
    p_value = float(p_value)

    if p_value <= alpha:
        line_label = f'{test_name} Statistic P-Value: p={p_value: 0.2f} <= {alpha}'
        verdict_label = 'Verdict: Reject $H_0$'

    else:
        line_label = f'{test_name} Statistic P-Value: p={p_value: 0.2f} > {alpha}'
        verdict_label = 'Verdict: Fail to Reject $H_0$'

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(f'Null Distribution Plot for {test_name}')
    n, bins, patches = ax.hist(null_dist, bins=100, density=True, alpha=0.7, label='Null Distribution', color='grey', edgecolor='black')

    ax.axvline(test_stat, color='blue', linestyle='--', linewidth=1.5, label=line_label)

    ax.text(
        test_stat,                          # x: aligned with the line
        ax.get_ylim()[1] * 0.5,             # y: middle of the y-axis
        f'p = {p_value:.2f}',
        color='blue',
        rotation=90,                        # make it vertical
        va='center', ha='right',            # vertical and horizontal alignment
        fontsize=10,
        backgroundcolor='white'             # optional: makes text readable over bars
    )

    if (test_stat > np.max(null_dist) * 1e2) or (test_stat < np.min(null_dist) * 1e2):
        ax.set_xscale('log')
        ax.minorticks_on()

    crit_value = np.quantile(null_dist, 1 - alpha)

    ax.axvspan(crit_value, ax.get_xlim()[1], color='red', alpha=0.3,
            label='Region for Statistical Significance (Reject H₀)')

    ax.axvline(crit_value, color='black', linestyle='-', linewidth=1.5, label='Beginning of Critical Value Region')

    # Add text label ON the shaded region
    ax.text(
        crit_value * 1.5,                # X position: move into the shaded region
        ax.get_ylim()[1] * 0.9,          # Y position: near top
        'Critical Region\n(Reject $H_0$)',
        color='darkred',
        fontsize=10,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)  # optional background
)

    title_label = ''
    if 'f' in test_name.lower():
        null_dist_label = 'Simulated F-type Mixture Null Distribution'
        if hypothesis == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: F-type test'
        elif hypothesis == "PAIRWISE":
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: F-type test'
    else:
        null_dist_label = r'Simulated $\chi^2_{1}$-type Mixture Null Distribution'
        if hypothesis == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: Squared L-2 Norm test'
        elif hypothesis== "PAIRWISE":
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: Squared L-2 Norm test'

    ax.set_ylabel('PDF', fontsize=14)
    ax.set_xlabel(null_dist_label, fontsize=14)
    ax.set_title(f'{title_label} \n{verdict_label}', fontsize=14)

    ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),  # Move it further down
    ncol=2,
    fontsize=12,
    frameon=False  # Optional: removes legend border)
    )

    ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show(block=False)

    return
