import numpy as np
import pandas as pd
import types
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats
from scipy.stats import chi2, ncx2, f
from scipy.linalg import inv, sqrtm

from functionalANOVA.timer import TimeBar, setUpTimeBar
from functionalANOVA.one_way import one_way
from functionalANOVA.one_way_BF import one_way_bf
from functionalANOVA.two_way import two_way
from functionalANOVA.two_way_BF import two_way_bf
from functionalANOVA.k_group_cov import k_group_cov
from functionalANOVA.k_group_cov_pairwise import k_group_cov_pairwise
from functionalANOVA.two_group_cov import two_group_cov
from functionalANOVA.function_subsetter import function_subsetter
from functionalANOVA.plot_means import plot_means
from functionalANOVA.plot_covariances import plot_covariances
from functionalANOVA.plot_test_stats import plot_test_stats


class functionalANOVA:

    def __init__(self, data_array, bounds_array, d_grid, subgroup_indicator, group_labels, primary_labels, secondary_labels, domain_label, response_label, domain_units_label, response_units_label, alpha, n_simul, n_boot, n_permutations, show_simul_plot, weights):

        self.data = data_array
        self.boundsArray = bounds_array
        self.d_grid = d_grid
        self.subgroup_indicator = subgroup_indicator

        self.group_labels = group_labels
        self.primary_labels = primary_labels
        self.secondary_labels = secondary_labels
        self.domain_label = domain_label
        self.response_label = response_label
        self.domain_units_label = domain_units_label
        self.response_units_label = response_units_label

        self.alpha = alpha
        self.n_simul = n_simul
        self.n_boot = n_boot
        self.n_permutations = n_permutations
        self.show_simul_plot = show_simul_plot
        self.weights = weights

        self.ANOVA_Methods = [
            "L2-Simul",
            "L2-Naive",
            "L2-BiasReduced",
            "L2-Bootstrap",
            "F-Simul",
            "F-Naive",
            "F-BiasReduced",
            "F-Bootstrap"
        ]

        self.COVAR_Methods = [
            "L2-Simul",
            "L2-Naive",
            "L2-BiasReduced",
            "Permutation-Test",
            "Bootstrap-Test"
        ]

        self.k_groups = len(self.data)
        self.n_i = [data.shape[1] for data in self.data]
        self.N = sum(self.n_i)
        self.n_domain_points = self.data[0].shape[0]

        if self.group_labels is None:
            self.group_labels = [f"Group {i+1}" for i in range(self.k_groups)]

        self.ANOVA_Methods_Used = []
        self.CriticalValues = []
        self.hypothesis = None
        self.hypothesis_LABEL = None

        self.SubgroupIndicator = self.subgroup_indicator
        self.PrimaryLabels = self.primary_labels
        self.SecondaryLabels = self.secondary_labels
        self.Weights = self.weights
        self.Hypothesis = None
        self.Contrast_Factor = None
        self.N_simul = self.n_simul
        self.N_boot = self.n_boot

        if self.group_labels is not None:
            self.generic_group_labels = (self.group_labels == [f"Group {i+1}" for i in range(self.k_groups)])
        else:
            self.generic_group_labels = True

    @staticmethod
    def chi_sq_mixture(df, coefs, N_samples):
        n_eigs = len(coefs)
        chi2rvs = np.random.chisquare(df, size=(n_eigs, N_samples))
        T_null = (coefs @ chi2rvs)
        return T_null

    @staticmethod
    def aflag_maker(n_i):
        aflag = []
        for k in range(len(n_i)):
            indicator = np.repeat(k + 1, n_i[k])
            aflag.extend(indicator)
        return np.array(aflag)


    def set_up_two_way(self):
        pass

    def generate_two_way_comb(self):
        pass

    def update_family_table(self, method, data):
        pass

    def l2_bootstrap(self, data):
        pass

    def f_bootstrap(self, data):
        pass

    def one_way(self, n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec):
        return one_way(self, n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec)

    def one_way_bf(self, method, data, contrast, c, indicator_a=None):
        return one_way_bf(self, method, data, contrast, c, indicator_a)

    def two_way(self, method, data, contrast):
        return two_way(self, method, data, contrast)

    def two_way_bf(self, method, data, contrast, c):
        return two_way_bf(self, method, data, contrast, c)

    def k_group_cov(self, method, stat, Sigma, V):
        return k_group_cov(self, method, stat, Sigma, V)

    def k_group_cov_pairwise(self, method, y1, y2):
        return k_group_cov_pairwise(self, method, y1, y2)

    def two_group_cov(self, method, y1, y2):
        return two_group_cov(self, method, y1, y2)

    def function_subsetter(self):
        return function_subsetter(self)

    def plot_means(self, plot_type='default', **kwargs):
        return plot_means(self, plot_type, **kwargs)

    def plot_covariances(self, plot_type='default', **kwargs):
        return plot_covariances(self, plot_type, **kwargs)

    def plot_test_stats(self, p_value, alpha, null_dist, test_stat, test_name, hypothesis, hypothesis_label):
        return plot_test_stats(p_value, alpha, null_dist, test_stat, test_name, hypothesis, hypothesis_label)

    def __str__(self):
        return f"functionalANOVA(k_groups={self.k_groups}, N={self.N}, n_domain_points={self.n_domain_points})"

    def __repr__(self):
        return self.__str__()
