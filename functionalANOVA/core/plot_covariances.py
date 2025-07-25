import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import math
from .utils import set_up_two_way, generate_two_way_comb


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
    fig_label = 'Group'
    temp_label = ''

    if not hasattr(self, 'subgroup_indicator') or self.subgroup_indicator is None or len(self.subgroup_indicator) == 0:
        assert plot_type == 'DEFAULT', 'TwoWay plotting options require a subgroup_indicator argument'

        if getattr(self, 'generic_group_labels', True):
            display_label = [f"Group {i+1}" for i in range(self.k_groups)]
        else:
            display_label = self.group_labels
    else:
        set_up_two_way(self)

        if plot_type in ['DEFAULT', 'PRIMARY']:
            plot_type = 'PRIMARY'
            fig_label = 'Primary Factor'
            display_label = self.primary_labels

            if getattr(self, 'generic_group_labels', True):
                display_label = [f"Group {label}" for label in display_label]

        elif plot_type == 'SECONDARY':
            fig_label = 'Secondary Factor'
            display_label = self.secondary_labels

            if getattr(self, 'generic_group_labels', True):
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

    if not self.response_units_label:
        color_bar_label = '(Response)^2'
    else:
        color_bar_label = f'({self.response_units_label})^2'

    if plot_type in ['DEFAULT', 'PRIMARY']:
        gamma_hat_i, pooled_covar, n_ii = _make_covariances(self.data, self.k_groups, self.n_domain_points)
        K = self.k_groups

    elif plot_type == 'SECONDARY':
        yy = np.concatenate([data.T for data in self.data], axis=0)

        bflag = self.subgroup_indicator
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
        aflag = np.repeat(np.arange(1, self.A_groups + 1), self.n_i)
        bflag = self.subgroup_indicator

        aflag0 = np.unique(aflag)
        bflag0 = np.unique(bflag)

        p, q = len(aflag0), len(bflag0)
        ab = p * q
        self.AB_groups = ab
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

    fig_width = position[2] / 100
    fig_height = position[3] / 100

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
            if not self.domain_units_label:
                temp_label = getattr(self, 'domain_label', '')
            else:
                temp_label = f"{getattr(self, 'domain_label', '')} ({self.domain_units_label})"
        else:
            if self.domain_units_label:
                temp_label = f"({self.domain_units_label})"

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
