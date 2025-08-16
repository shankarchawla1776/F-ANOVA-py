import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os
import math
from functionalANOVA.core import utils
from typing import Union, Optional, Any, Tuple
from matplotlib.colors import Colormap, ListedColormap
from dataclasses import dataclass


class PlotOptions:
    # Class-level constant of valid legend locations
    _VALID_LEGEND_LOCATIONS = [
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        "best",
    ]

    def __init__(
        self,
        observation_size_label: bool=True,
        x_scale: str = "linear",
        y_scale: str = "linear",
        data_transparency: float = 0.1,
        legend_transparency: float = 0.3333,
        data_line_width: float = 1.75,
        mean_line_width: float = 5,
        font_size: int = 18,
        title_labels: Optional[Any] = None,
        save_path: str = "",
        legend_location: str = "best",
        num_columns: int = 1,
        legend_title: str = "",
        new_colors: Optional[np.ndarray] = None,
        position: Tuple[int, int, int, int] = (90, 90, 1400, 800),
    ):
        
        self.observation_size_label = observation_size_label
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.data_transparency = data_transparency
        self.legend_transparency = legend_transparency
        self.data_line_width = data_line_width
        self.mean_line_width = mean_line_width
        self.font_size = font_size
        self.title_labels = title_labels
        self.save_labels = title_labels
        self.save_path = save_path
        self.legend_location = legend_location
        self.num_columns = num_columns
        self.legend_title = legend_title
        self.new_colors = new_colors
        self.position = position

        self._validate()

    def _validate(self):
        # scales
        valid_scales = ("linear", "log", "symlog", "logit")
        if self.x_scale not in valid_scales:
            raise ValueError(f"x_scale must be one of {valid_scales}, got {self.x_scale!r}")
        if self.y_scale not in valid_scales:
            raise ValueError(f"y_scale must be one of {valid_scales}, got {self.y_scale!r}")

        # alpha-like params
        if not (0.0 <= self.data_transparency <= 1.0):
            raise ValueError(f"data_transparency must be in [0, 1], got {self.data_transparency}")
        if not (0.0 <= self.legend_transparency <= 1.0):
            raise ValueError(f"legend_transparency must be in [0, 1], got {self.legend_transparency}")

        # legend options
        if not isinstance(self.legend_title, str):
            raise TypeError(f"legend_title must be a string, got {type(self.legend_title).__name__}")
        if self.legend_location not in self._VALID_LEGEND_LOCATIONS:
            raise ValueError(
                f"legend_location must be one of {self._VALID_LEGEND_LOCATIONS}, got {self.legend_location!r}"
            )

        # layout options
        if not isinstance(self.num_columns, int) or self.num_columns < 1:
            raise ValueError("num_columns must be an integer >= 1")
        if not (isinstance(self.position, tuple) and len(self.position) == 4 and all(isinstance(v, int) for v in self.position)):
            raise ValueError("position must be a tuple of 4 integers, e.g. (x, y, width, height)")

        # save_path: if provided, it must be an existing directory
        if isinstance(self.save_path, str) and self.save_path != "":
            normalized = os.path.abspath(os.path.expanduser(self.save_path))
            if not os.path.isdir(normalized):
                raise ValueError(f"save_path must be an existing directory, got {self.save_path!r}")
            self.save_path = normalized  # normalize
            
    def update_from_dict(self, updates: dict) -> None:
        """
        Update PlotOptions from a dictionary. 
        Only applies changes if value is not None and different.
        """
        for key, value in updates.items():
            if value is not None and hasattr(self, key):
                if getattr(self, key) != value:
                    setattr(self, key, value)
        self._validate()


# TODO: Cleaner than passing tons of args in. Need to verify twoway plotting
def plot_means(self, plot_type):

    plot_type = plot_type.upper()
    n_labels = []
    the_labels  = []


    if self._groups.subgroup_indicator is None:
        the_labels = self._labels.group
        n_labels = self.n_i
    else:
        self._setup_twoway(self)

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
            the_labels = utils.generate_two_way_comb(self)
            n_labels = np.concatenate([item for sublist in self.n_ii for item in sublist])

    if self.plottingOptions.observation_size_label:
        if self._labels.generic_group:
            the_data_labels = [f": Group Data ({int(n)})" for n in n_labels]
        else:
            the_data_labels = [f": Data ({int(n)})" for n in n_labels]
    else:
        the_data_labels = [''] * len(n_labels)

    if self._labels.generic_group:
        mean_group_labels = [f"{label}: Group Mean" for label in the_labels]
        data_group_labels = [f"{the_labels[i]}{the_data_labels[i]}" for i in range(len(the_labels))]
    else:
        mean_group_labels = [f"{label}: Mean" for label in the_labels]
        data_group_labels = [f"{the_labels[i]}{the_data_labels[i]}" for i in range(len(the_labels))]


    # fast patch for figure scale errors
    fig_width = max(self.plottingOptions.position[2] / 100, 8)
    fig_height = max(self.plottingOptions.position[3] / 100, 6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    match plot_type:
        case 'DEFAULT':
            mean_groups = [np.mean(data, axis=1) for data in self.data]

            if self.plottingOptions.new_colors is None:
                if self._groups.k < 7:
                    color_list = plt.cm.tab10(np.linspace(0, 1, self._groups.k))
                else:
                    color_list = plt.cm.turbo(np.linspace(0, 1, self._groups.k))
            else:
                assert self.plottingOptions.new_colors.shape[1] == 4, 'Color order matrix must have 4 columns representing RGB values from 0 to 1 with the last column being alpha the transparency'
                assert self.plottingOptions.new_colors.shape[0] >= self._groups.k, f'Color order matrix must have at least {self._groups.k} rows representing all the groups'
                color_list = self.plottingOptions.new_colors

            plot_reals = []
            legend_example_lines = []

            for k in range(self._groups.k):
                lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.data_transparency)
                plot_reals.append(lines)

                legend_line = Line2D([0], [0], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.legend_transparency)
                legend_example_lines.append(legend_line)

            plot_means = []

            for k in range(self._groups.k):
                line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k],
                            linewidth=self.plottingOptions.mean_line_width, linestyle='--')
                plot_means.append(line)

            ax.set_title(f'F-ANOVA Group Means and Realizations {self.plottingOptions.title_labels}', fontsize=self.plottingOptions.font_size)
            lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=self.plottingOptions.legend_location, ncol=self.plottingOptions.num_columns)

            if self.plottingOptions.legend_title:
                lg.set_title(self.plottingOptions.legend_title)

        case 'PRIMARY':
            mean_groups = [np.mean(data, axis=1) for data in self.data]

            if self.plottingOptions.new_colors is None:
                if self._groups.k < 7:
                    color_list = plt.cm.tab10(np.linspace(0, 1, self._groups.k))
                else:
                    color_list = plt.cm.turbo(np.linspace(0, 1, self._groups.k))
            else:
                color_list = self.plottingOptions.new_colors

            plot_reals = []
            legend_example_lines = []

            for k in range(self._groups.k):
                lines = ax.plot(self.d_grid, self.data[k], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.data_transparency)
                plot_reals.append(lines)

                legend_line = Line2D([0], [0], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.legend_transparency)
                legend_example_lines.append(legend_line)

            plot_means = []

            for k in range(self._groups.k):
                line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k], linewidth=self.plottingOptions.mean_line_width, linestyle='--')
                plot_means.append(line)

            ax.set_title(f'F-ANOVA Primary Factor Means and Realizations {self.plottingOptions.title_labels}', fontsize=self.plottingOptions.font_size)
            lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=self.plottingOptions.legend_location, ncol=self.plottingOptions.num_columns)

            if not self.plottingOptions.legend_title:
                lg.set_title("Primary Factor Levels")
            else:
                lg.set_title(self.plottingOptions.legend_title)

        case 'SECONDARY':
            yy = np.concatenate([data.T for data in self.data], axis=0)

            bflag = self._groups.subgroup_indicator
            bflag0 = np.unique(bflag)
            n_secondary = len(bflag0)

            if self.plottingOptions.new_colors is None:
                if n_secondary < 7:
                    color_list = plt.cm.tab10(np.linspace(0, 1, n_secondary))
                else:
                    color_list = plt.cm.turbo(np.linspace(0, 1, n_secondary))
            else:
                color_list = self.plottingOptions.new_colors

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
                lines = ax.plot(self.d_grid, sub_data[k], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.data_transparency)
                plot_reals.append(lines)

                legend_line = Line2D([0], [0], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.legend_transparency)
                legend_example_lines.append(legend_line)

            plot_means = []

            for k in range(n_secondary):
                line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k],
                            linewidth=self.plottingOptions.mean_line_width, linestyle='--')
                plot_means.append(line)

            ax.set_title(f'F-ANOVA Secondary Factor Means and Realizations {self.plottingOptions.title_labels}', fontsize=self.plottingOptions.font_size)
            lg = ax.legend(plot_means + legend_example_lines, mean_group_labels + data_group_labels, loc=self.plottingOptions.legend_location, ncol=self.plottingOptions.num_columns)

            if not self.plottingOptions.legend_title:
                lg.set_title("Secondary Factor Levels")
            else:
                lg.set_title(self.plottingOptions.legend_title)

        case 'INTERACTION':
            yy = np.concatenate([data.T for data in self.data], axis=0)

            aflag = np.repeat(np.arange(1, self._groups.A + 1), self.n_i)
            bflag = self._groups.subgroup_indicator
            aflag0 = np.unique(aflag)
            bflag0 = np.unique(bflag)
            p, q = len(aflag0), len(bflag0)
            ab = p * q

            combinations = utils.generate_two_way_comb(self)
            mean_labels = [f"{comb}: Mean" for comb in combinations]

            if self.plottingOptions.observation_size_label:
                data_labels = [f"{combinations[k]}: Data ({int(n_labels[k])})" for k in range(len(combinations))]
            else:
                data_labels = [f"{comb}: Data" for comb in combinations]

            if self.plottingOptions.new_colors is None:
                if ab < 7:
                    color_list = plt.cm.tab10(np.linspace(0, 1, ab))
                else:
                    color_list = plt.cm.turbo(np.linspace(0, 1, ab))
            else:
                color_list = self.plottingOptions.new_colors

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
                lines = ax.plot(self.d_grid, sub_data[k], color=color_list[k], linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.data_transparency)
                plot_reals.append(lines)

                legend_line = Line2D([0], [0], color=color_list[k],linewidth=self.plottingOptions.data_line_width, alpha=self.plottingOptions.legend_transparency)
                legend_example_lines.append(legend_line)

            plot_means = []

            for k in range(ab):
                line, = ax.plot(self.d_grid, mean_groups[k], color=color_list[k], linewidth=self.plottingOptions.mean_line_width, linestyle='--')
                plot_means.append(line)

            ax.set_title(f'F-ANOVA Primary and Secondary Factor Combinatorial Means and Realizations {self.plottingOptions.title_labels}', fontsize=self.plottingOptions.font_size)
            lg = ax.legend(plot_means + legend_example_lines, mean_labels + data_labels, loc=self.plottingOptions.legend_location, ncol=self.plottingOptions.num_columns)

            if not self.plottingOptions.legend_title:
                lg.set_title("TwoWay Factor Levels")
            else:
                lg.set_title(self.plottingOptions.legend_title)


    ax.set_xscale(self.plottingOptions.x_scale)
    ax.set_yscale(self.plottingOptions.y_scale)


    ax.set_xlabel(auto_make_labels(self._units.domain, self._labels.domain), fontsize=self.plottingOptions.font_size)
    ax.set_ylabel(auto_make_labels(self._units.response, self._labels.response), fontsize=self.plottingOptions.font_size)

    ax.tick_params(labelsize=self.plottingOptions.font_size)

    if self.plottingOptions.save_path:

        save_filename = f"GroupMeans_{self.plottingOptions.save_labels}.png"

        fig.savefig(os.path.join(self.plottingOptions.save_path, save_filename))
        plt.close(fig)
    else:
        plt.show(block=False)

    return fig, ax


def auto_make_labels(units, quanity_label):
    if quanity_label is not None:
        axis_label = f'{quanity_label}'
    else:
        axis_label = ''
        
    if units is not None:
        if axis_label == '':
            axis_label = f'({units})'
        else:
            axis_label += ' ' + f'({units})'

    return axis_label
       


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
        plt.show(block=False)

    return fig

# TODO: Some option or ability to save these plots
def plot_test_stats(self, p_value:float | np.floating, null_dist:np.ndarray, test_stat:float | np.floating,
                    test_name:str, scedasticity:str,  k:int, N:int|None = None):
    
    p_value = float(p_value)

    if p_value <= self.alpha:
        line_label = f'{test_name} Statistic P-Value: p={p_value: 0.2f} <= {self.alpha}'
        verdict_label = 'Verdict: Reject $H_0$'

    else:
        line_label = f'{test_name} Statistic P-Value: p={p_value: 0.2f} > {self.alpha}'
        verdict_label = 'Verdict: Fail to Reject $H_0$'

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(f'Null Distribution Plot for {test_name}')
    n, bins, patches = ax.hist(null_dist, bins=100, density=True, alpha=0.7, label='Null Distribution', color='grey', edgecolor='black')

    ax.axvline(float(test_stat), color='blue', linestyle='--', linewidth=1.5, label=line_label)

    ax.text(
        float(test_stat),                          # x: aligned with the line
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

    crit_value = np.quantile(null_dist, 1 - self.alpha)

    x_max = max(np.max(null_dist), test_stat) * 1.2  # ensure room for shading
    ax.set_xlim(right=x_max)

    ax.axvspan(crit_value, x_max, color='red', alpha=0.3,
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
        assert N is not None and isinstance(N, int), "'N' must be provided for F-type mixture labeling"
        
        null_dist_label = fr'Simulated $F_{{(k-1),(n-k)}} = F_{{({k - 1}),({N - k})}}$-type Mixture Null Distribution'
        if self.hypothesis == "FAMILY":
            title_label = f'One-Way({scedasticity}), Family, Functional ANOVA: F-type test'
        elif self.hypothesis == "PAIRWISE":
            title_label = (f"One-Way({scedasticity}), Pairwise ({self._labels.hypothesis})\n" r"Functional ANOVA: F-type test")
    else:
        null_dist_label = fr'Simulated $\chi^2_{{(k-1)}} = \chi^2_{{{k - 1}}}$-type Mixture Null Distribution'
        if self.hypothesis == "FAMILY":
            title_label = fr'One-Way ({scedasticity}), Family, Functional ANOVA: Squared $L^2$ Norm test'
        elif self.hypothesis== "PAIRWISE":
            title_label = (f"One-Way ({scedasticity}), Pairwise ({self._labels.hypothesis})\n"
                           r"Functional ANOVA: F-type test")


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
