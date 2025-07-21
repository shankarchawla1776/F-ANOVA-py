import numpy as np
import pandas as pd
import types
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats
from scipy.stats import chi2, ncx2, f
from scipy.linalg import inv, sqrtm
from .utils import chi_sq_mixture

from concurrent.futures import ProcessPoolExecutor, as_completed
import pytest

from .timer import TimeBar, set_up_time_bar
from .plot_test_stats import plot_test_stats
from .utils import l2_bootstrap, f_bootstrap, update_family_table

def one_way(self, n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec):

    def _group_booter(data_matrix_cell, n_domain_points, k_groups, n_i, n):
        eta_i_star = np.zeros((n_domain_points, k_groups))
        build_covar_star = np.zeros((n_domain_points, 0))

        for k in range(k_groups):
            indices = np.random.choice(data_matrix_cell[k].shape[1], n_i[k], replace=True)
            data_subset_boot = data_matrix_cell[k][:,indices]

            eta_i_star[:,k] = np.mean(data_subset_boot, axis=1)

            zero_mean_data_k_subset = data_subset_boot - eta_i_star[:,k].reshape(-1,1)
            build_covar_star = np.hstack((build_covar_star, zero_mean_data_k_subset))

        gamma_hat_star = (1 / (n-k_groups)) * (build_covar_star.T @ build_covar_star)
        eta_grand_star = np.sum(eta_i_star * n_i, axis=1) / n

        return eta_i_star, eta_grand_star, gamma_hat_star

    n_methods = len(self.ANOVA_Methods_Used)

    pvalue_matrix = np.zeros((n_tests, n_methods))

    self.CriticalValues = [[None, None, None] for _ in range(n_methods)]

    T_n, F_n, beta_hat, kappa_hat, beta_hat_unbias, kappa_hat_unbias, C, D = params

    counter = 0

    for method in self.ANOVA_Methods_Used:
        counter += 1

        if method == "L2-Simul":
            T_null = chi_sq_mixture(q, eig_gamma_hat, self.n_simul)

            T_NullFitted = stats.gaussian_kde(T_null)

            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - T_NullFitted.integrate_box_1d(-np.inf, T_n[j])
                p_value[j] = max(0,min(1,p_value[j]))

                if self.show_simul_plot:
                    plot_test_stats(p_value[j], self.alpha, T_null, T_n[j], method + " test", self.hypothesis, pair_vec[j])

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][1] = np.quantile(T_null, 1-self.alpha)

            if self.hypothesis == "FAMILY":
                update_family_table(self, method, [T_NullFitted])

        elif method == "L2-Naive":
            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - ncx2.cdf(T_n[j] / beta_hat, q * kappa_hat, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][1] = beta_hat * ncx2.ppf(1 - self.alpha, q * kappa_hat, 0)

        elif method == "L2-BiasReduced":
            p_value = np.zeros((n_tests,1))
            for j in range(n_tests):
                p_value[j] = 1 - ncx2.cdf(T_n[j] / beta_hat_unbias, q * kappa_hat_unbias, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][1] = beta_hat_unbias * ncx2.ppf(1 - self.alpha, q * kappa_hat_unbias,0)

        elif method == "L2-Bootstrap":
            T_n_Boot = np.zeros((self.n_boot, n_tests))

            if self.hypothesis == "FAMILY":
                self.hypothesis_LABEL = pair_vec[0]

                yy = np.array([])

                for j in range(self.k_groups):
                    # yy = np.append(yy, self.data[j].T)
                    if j == 0:
                        yy = self.data[j].T
                    else:
                        yy = np.vstack([yy, self.data[j].T])

                # _, _, T_n_Boot[:,0] = self.L2Bootstrap(self, yy)
                _, _, T_n_Boot[:,0] = l2_bootstrap(self, yy)

                update_family_table(self, method, [self.n_boot])

            elif self.hypothesis == "PAIRWISE":
                # n_tests is the number of tests; n = self.N is real n
                for j in range(n_tests):

                    self.hypothesis_LABEL = pair_vec[j]
                    T = set_up_time_bar(method)

                    Ct = C[j, :]

                    d_points = self.n_domain_points
                    k_group = self.k_groups
                    n_iii = self.n_i
                    g_data = self.data
                    n = self.N

                    for k in range(self.n_boot):
                        eta_i_star, _, _ = _group_booter(g_data, d_points, k_group, n_iii, n)
                        SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * inv(Ct @ D @ Ct.T)
                        T_n_Boot[k,j] = np.sum(SSH_t)
                        T.progress()

                    T.stop()
                    T.delete()

            p_value = np.zeros((n_tests, 1))
            crit_vals = np.zeros((n_tests, 1))
            for j in range(n_tests):
                p_value[j] = np.mean(T_n_Boot[:,j] >= T_n[j])
                crit_vals[j] = np.quantile(T_n_Boot[:,j], 1 - self.alpha)

            pvalue_matrix[:,counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][1] = crit_vals[-1]


        elif method == "F-Simul":

            ratio = (self.N - self.k_groups) / q
            T_null = chi_sq_mixture(q, eig_gamma_hat, self.n_simul)
            F_null_denom = chi_sq_mixture(self.N - self.k_groups, eig_gamma_hat, self.n_simul)
            F_null = (T_null / F_null_denom) * ratio
            F_NullFitted = stats.gaussian_kde(F_null)

            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - F_NullFitted.integrate_box_1d(-np.inf, F_n[j])
                p_value[j] = max(0,min(1,p_value[j]))
                if self.show_simul_plot:
                    plot_test_stats(p_value[j], self.alpha, F_null, F_n[j], method + " test", self.hypothesis, pair_vec[j])

            pvalue_matrix[:, counter-1] = p_value.flatten()

            if self.hypothesis == "FAMILY":
                update_family_table(self, method, [F_NullFitted])

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][2] = np.quantile(F_null, 1 - self.alpha)
            self.CriticalValues[counter-1][1] = np.quantile((self.CriticalValues[counter-1][2] / ratio) * F_null_denom, 1 - self.alpha)

        elif method == "F-Naive":
            p_value = np.zeros((n_tests,1))
            for j in range(n_tests):
                p_value[j] = 1 - f.cdf(F_n[j], q * kappa_hat, (self.N - self.k_groups) * kappa_hat)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            A = q * kappa_hat
            B = (self.N - self.k_groups) * kappa_hat

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][2] = f.ppf(1 - self.alpha, A, B)

        elif method == "F-BiasReduced":
            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - f.cdf(F_n[j], q * kappa_hat_unbias, (self.N - self.k_groups) * kappa_hat_unbias)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][2] = f.ppf(1 - self.alpha, q * kappa_hat_unbias, (self.N - self.k_groups) * kappa_hat_unbias)

        elif method == "F-Bootstrap":
            F_n_Boot = np.zeros((self.n_boot, n_tests))
            ratio = (self.N - self.k_groups) / q
            crit_vals = np.zeros((n_tests, 1))
            ReversedT_n = np.zeros((n_tests, 1))

            if self.hypothesis == "FAMILY":
                self.hypothesis_LABEL = pair_vec[0]
                f_n_Denominator_Boot = np.nan * np.ones((self.n_boot, n_tests))
                yy = np.array([])

                for j in range(self.k_groups):
                    # yy = np.append(yy, self.data[j].T)

                    if j == 0:
                        yy = self.data[j].T
                    else:
                        yy = np.vstack([yy, self.data[j].T])

                # _, _, F_n_Boot[:,0] = self.FBootstrap(self, yy)
                _, _, F_n_Boot[:,0] = f_bootstrap(self, yy)

                update_family_table(self, method, [self.n_boot])

            elif self.hypothesis == "PAIRWISE":
                f_n_Denominator_Boot = np.zeros((self.n_boot, n_tests))

                for j in range(n_tests):
                    self.hypothesis_LABEL = pair_vec[j]
                    T = set_up_time_bar(method)
                    Ct = C[j, :]

                    d_points = self.n_domain_points
                    k_group = self.k_groups
                    n_iii = self.n_i
                    g_data = self.data
                    n = self.N

                    for k in range(self.n_boot):
                        eta_i_star, _, gamma_hat_star = _group_booter(g_data, d_points, k_group, n_iii, n)
                        f_n_Denominator_Boot[k,j] = np.trace(gamma_hat_star) * (n - k_group)
                        SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * inv(Ct @ D @ Ct.T)

                        T_n_Boot = np.sum(SSH_t)
                        F_n_Boot[k,:] = (T_n_Boot / f_n_Denominator_Boot[k,j]) * ratio
                        T.progress()

                    T.stop()
                    T.delete()

            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - np.sum(F_n_Boot[:,j] < F_n[j]) / float(self.n_boot)

                crit_vals[j] = np.quantile(F_n_Boot[:,j], 1 - self.alpha)
                ReversedT_n[j] = np.quantile((crit_vals[j] / ratio) * f_n_Denominator_Boot[:,j], 1 - self.alpha)

            pvalue_matrix[:,counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = method
            self.CriticalValues[counter-1][1] = ReversedT_n[j]
            self.CriticalValues[counter-1][2] = crit_vals[j]


    return pvalue_matrix
