import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
from .utils import set_up_two_way, generate_two_way_comb


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
        self.subgroup_indicator = subgroup_indicator
    if group_labels is not None:
        self.group_labels = group_labels
    if primary_labels is not None:
        self.primary_labels = primary_labels
    if secondary_labels is not None:
        self.secondary_labels = secondary_labels
    if domain_units_label:
        self.domain_units_label = domain_units_label
    if response_units_label:
        self.response_units_label = response_units_label

    plot_type = plot_type.upper()

    domain_label = getattr(self, 'domain_label', '')
    response_label = getattr(self, 'response_label', '')

    if not hasattr(self, 'subgroup_indicator') or self.subgroup_indicator is None or len(self.subgroup_indicator) == 0:

        assert plot_type == 'DEFAULT', 'TwoWay plotting options require a subgroup_indicator argument'
        the_labels = self.group_labels
        n_labels = self.n_i
    else:
        set_up_two_way(self)

        if plot_type in ['DEFAULT', 'PRIMARY']:
            plot_type = 'PRIMARY'
            the_labels = self.primary_labels
            n_labels = self.n_i

        elif plot_type == "SECONDARY":
            the_labels = self.secondary_labels
            n_labels = np.zeros(self.B_groups)
            for k in range(self.A_groups):
                for kk in range(self.B_groups):
                    n_labels[kk] += self.n_ii[k][kk]

        elif plot_type == "INTERACTION":
            the_labels = generate_two_way_comb(self)
            n_labels = np.concatenate([item for sublist in self.n_ii for item in sublist])

    if observation_size_label:
        if getattr(self, 'generic_group_labels', True):
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

    fig_width = position[2] / 100
    fig_height = position[3] / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if plot_type == 'DEFAULT':
        mean_groups = [np.mean(data, axis=1) for data in self.data]

        if new_colors is None:
            if self.k_groups < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, self.k_groups))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, self.k_groups))
        else:
            assert new_colors.shape[1] == 3, 'Color order matrix must have 3 columns representing RGB values from 0 to 1'
            assert new_colors.shape[0] >= self.k_groups, f'Color order matrix must have at least {self.k_groups} rows representing all the groups'
            color_list = new_colors

        plot_reals = []
        legend_example_lines = []

        for k in range(self.k_groups):
            lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k], linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(self.k_groups):
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
            if self.k_groups < 7:
                color_list = plt.cm.tab10(np.linspace(0, 1, self.k_groups))
            else:
                color_list = plt.cm.turbo(np.linspace(0, 1, self.k_groups))
        else:
            color_list = new_colors

        plot_reals = []
        legend_example_lines = []

        for k in range(self.k_groups):
            lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=data_line_width, alpha=data_transparency)
            plot_reals.append(lines)

            legend_line = Line2D([0], [0], color=color_list[k], linewidth=data_line_width, alpha=legend_transparency)
            legend_example_lines.append(legend_line)

        plot_means = []

        for k in range(self.k_groups):
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

        bflag = self.subgroup_indicator
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

        aflag = np.repeat(np.arange(1, self.A_groups + 1), self.n_i)
        bflag = self.subgroup_indicator
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
        if self.domain_units_label:
            ax.set_xlabel(f'{domain_label} ({self.domain_units_label})', fontsize=font_size)
        else:
            ax.set_xlabel(domain_label, fontsize=font_size)

        if self.response_units_label:
            ax.set_ylabel(f'{response_label} ({self.response_units_label})', fontsize=font_size)
        else:
            ax.set_ylabel(response_label, fontsize=font_size)
    else:
        if self.domain_units_label:
            ax.set_xlabel(f'({self.domain_units_label})', fontsize=font_size)

        if self.response_units_label:
            ax.set_ylabel(f'({self.response_units_label})', fontsize=font_size)

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
