from typing import T
from test.test_logging import ZERO
from tracemalloc import DomainFilter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pandas._config import display
import seaborn as sns
from matplotlib.gridspec import GridSpec
from utils import *


def plot_covariances(self, plot_type='default', subgroup_indicator=None, group_labels=None, primary_labels=None, secondary_labels=None, x_scale='', y_scale='', color_scale='', domain_units_label='', response_units_label='', title_labels=None, save_path='', position=(90, 257, 2000, 800)):
    if subgroup_indicator is None:
        subgroup_indicator = getattr(self, 'SubgroupIndicator', None)
    if group_labels is None:
        group_labels = getattr(self, 'GroupLabels', [])
    if primary_labels is None:
        primary_labels = getattr(self, 'PrimaryLabels', [])
    if secondary_labels is None:
        secondary_labels = getattr(self, 'SecondaryLabels', [])
    if domain_units_label == '':
        domain_units_label = getattr(self, 'domainUnitsLabel', '')
    if response_units_label == '':
        response_units_label = getattr(self, 'responseUnitsLabel', '')

    plot_type = plot_type.upper()
    fig_label = 'Group'
    temp_label = ''


    if subgroup_indicator is None or len(subgroup_indicator) == 0:
        assert plot_type = 'DEFAULT', "TwoWay plotting options require a SubgroupIndicator argument"

        if self.generic_group_labels:
            display_label = [f"Group {i+1}" for i in range(self.k_groups)]
        else:
            display_label = group_labels


    else:
        self.set_up_to_way()

        if plot_type in ['DEFAULT', 'PRIMARY']:
            plot_type = 'PRIMARY'
            fig_label = 'Primary Factor'
            display_label = primary_labels
            if self.generic_group_labels:
                display_label = [f"Group {label}" for label in display_label]
        elif plot_type == "SECONDARY":
            fig_label = 'Secondary Factor'
            display_label = secondary_labels
            if self.generic_group_labels:
                display_label = [f"Group {label}" for label in display_label]
        elif plot_type == "INTERACTION":
            fig_label = 'Primary & Secondary Factor'
            combinations = generate_two_way_comb(self)
            display_label = combinations

    if hasattr(self, 'EchoEnsembleRecs') and self.EchoEnsembleRecs is not None:
        if title_labels:
            title_labels_str = self.EchoEnsembleRecs.make_summary_string(title_labels, True)
            save_labels = self.EchoEnsembleRecs.make_summary_string(title_labels, True, sanitize_string=True)
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


    if response_units_label == '':
        color_bar_label = '(Reponse)^2'
    else:
        color_bar_label = f"({response_units_label})^2"

    if plot_type in ['DEFAULT', 'PRIMARY']:
        gamma_hat_i, pooled_covar, n_ii = make_covariances(self.data, self.k_groups,self.n_domain_points)
        K = self.k_groups


    elif plot_type == "SECONDARY":
        yy = np.vstack([i.T for i in self.data])
        bflag = subgroup_indicator
        bflag0 = np.unique(subgroup_indicator)
        N_secondary = len(bflag0)
        sub_data = []

        for k in range(N_secondary):
            flag = bflag == bflag0[k]
            sub_data.append(yy[flag, :].T)

        gamma_hat_i, pooled_covar, n_ii = make_covariances(sub_data, N_secondary, self.n_domain_points)
        K = N_secondary

    elif plot_type == "INTERACTION":
        yy = np.vstack([i.T for i in self.data])
        aflag = aflag_maker(self.n_i)
        bflag = subgroup_indicator
        aflag0 = np.unique(aflag)
        p = len(aflag0)
        bflag0 = np.unique(subgroup_indicator)
        q = len(bflag0)
        ab = p * q
        self.AB_groups = ab

        sub_data = []
        counter = 0
        for i in range(p):
            for j in range(q):
                ijflag = (aflag == aflag0[i]) & (bflag == bflag0[i])
                yyi = yy[ijflag, :]
                sub_data.append(yyi.T)
                counter += 1

        gamma_hat_i, pooled_cover, n_ii = make_covariances(sub_data, ab, self.n_domain_points)
        K = ab

    n_plots = K + 1
    rows, cols = grid_generator(n_plots)

    fig = plt.figure(figsize=(position[2]/100, position[3]/100))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)


    for ii in range(K):
        ax = fig.add_subplot(gs[ii // cols, ii % cols])

        cov_matrix = gamma_hat_i[:, :, ii]
        cmin = np.min(cov_matrix)
        cmax = np.max(cov_matrix)

        im = ax.imshow(cov_matrix, cmap='viridis', aspect='equal', extent=[self.d_grid[0], self.d_grid[-1], self.d_grid[0], self.d_grid[-1]],vmin=cmin, vmax=cmax, origin='lower')

        ax.set_title(f"{display_label[ii]} | n = {n_ii[ii]}")

        if hasattr(self, 'EchoEnsembleRecs') and self.EchoEnsembleRecs is not None:
            if domain_units_label == '':
                temp_label = self.domain_label
            else:
                temp_label = f"{self.domain_label} ({domain_units_label})"
        else:
            if domain_units_label != '':
                temp_label = f"({domain_units_label})"

        ax.set_xlabel(temp_label)
        ax.set_ylabel(temp_label)

        if x_scale == '':
            ax.set_xscale('linear')
        else:
            ax.set_xscale(x_scale)

        if y_scale == '':
            ax.set_yscale('linear')
        else:
            ax.set_yscale(y_scale)

        plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[-1, -1])
    cmin = np.min(pooled_covar)
    cmax = np.max(pooled_covar)

    im = ax.imshow(pooled_covar, cmap='viridis', aspect='equal',extent=[self.d_grid[0], self.d_grid[-1], self.d_grid[0], self.d_grid[-1]],vmin=cmin, vmax=cmax, origin='lower')

    ax.set_title(f"Pooled | n = {self.N}")
    ax.set_xlabel(temp_label)
    ax.set_ylabel(temp_label)
    plt.colorbar(im, ax=ax)

    for ax in fig.get_axes():
        ax.tick_params(labelsize=18)

    fig.suptitle(f'{fig_label} Covariances\n{title_labels_str}', fontsize=20)

    if save_path:
        save_labels = f"Covariance_{save_labels}"
        fig.savefig(f"{save_path}/{save_labels}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()





def make_covariances(data, K, domain_points):

    eta_i = np.zeros((domain_points, K))
    v_hat_i = [None] * K
    gamma_hat_i = np.zeros((domain_points, domain_points, K))
    n_ii = np.zeros(K)

    for kk in range(K):
        n_i = data[kk].shape[1]
        n_ii[kk] = n_i
        eta_i[:, kk] = np.mean(data[kk], axis=1)
        zero_mean_data_k_subset = data[kk] - eta_i[:, kk].reshape(-1,1)
        v_hat_i[kk] = zero_mean_data_k_subset
        gamma_hat_i[:, :, kk] = (1 / (n_i -1)) * (zero_mean_data_k_subset @ zero_mean_data_k_subset.T)

    N = np.sum(n_ii)

    pooled_covar_terms = np.zeros((domain_points, domain_points, K))
    for kk in range(K):
        pooled_covar_terms[:, :, kk] = (n_ii[kk] -1) * gamma_hat_i[:,:,kk]

    pooled_covar = np.sum(pooled_covar_terms, axis=2) / (N-K)

    return gamma_hat_i, pooled_covar, n_ii.astype(int)


def grid_generator(K):
    if K<=3:
        N=K
        M=1
    elif K==4:
        N=2
        M=2
    elif K==5:
        N=2
        M=3
    elif K==6:
        N=2
        M=3
    elif K==7:
        N=3
        M=3
    elif K==8:
        N=3
        M=3
    else:
        N = int(np.ceil(np.sqrt(K)))
        M = int(np.ceil(K/N))

    return M, N


def generate_two_way_comb(self):
    combinations = []
    counter = 0
    for k in range(self.A_groups):
        for kk in range(self.B_groups):
            combinations.append(f"{self.PrimaryLabels[k]}-{self.SecondaryLabels[kk]}")
            counter += 1

    return combinations
