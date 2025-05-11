import numpy as np
import pandas as pd
import types
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats
from scipy.stats import chi2, ncx2, f
from scipy.linalg import inv, sqrtm

from concurrent.futures import ProcessPoolExecutor, as_completed
import pytest

from Utils import *

def one_way(self, n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec):

    n_methods = len(self.ANOVA_Methods_Used)

    pvalue_matrix = np.zeros((n_tests, n_methods))

    self.CriticalValues = [[None, None, None] for _ in range(n_methods)]

    T_n, F_n, beta_hat, kappa_hat, beta_hat_unibas, kappa_hat_unbias, C, D = params

    counter = 0

    for i in self.ANOVA_Methods_Used:
        counter += 1

        if i == "L2-Simul":
            T_null = chi_sq_mixture(q, eig_gamma_hat, self.N_simul)

            T_NullFitted = stats.gaussian_kde(T_null)

            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - T_NullFitted.integrate_box_1d(-np.inf, T_n[j])

                if self.showSimulPlot:
                    plot_test_stats(p_value[j], self.alpha, T_null, T_n[j], i + " test", self.Hypothesis, pair_vec[j])

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = i
            self.CriticalValues[counter-1][1] = np.quantile(T_null, 1-self.alpha)

            if self.Hypothesis == "FAMILY":
                self.update_family_table(i, [T_NullFitted])

        elif i == "L2-Naive":
            p_value = np.zeros((n_tests, 1))

            for j in range(n_tests):
                p_value[j] = 1 - ncx2.cdf(T_n[j] / beta_hat, q * kappa_hat, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = i
            self.CriticalValues[counter-1][1] = beta_hat * ncx2.ppf(1 - self.alpha, q * kappa_hat, 0)

            # self. = ==a sd
        elif i == "L2-BiasReduced":
            p_value = np.zeros((n_tests,1))
            for j in range(n_tests):
                p_value[j] = 1 - ncx2.cdf(T_n[j] * beta_hat_unibas, q * kappa_hat_unbias, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter -1][0] = i
            self.CriticalValues[counter-1][1] = beta_hat_unibas * ncx2.ppf(1 - self.alpha, q * kappa_hat_unbias,0)

        elif i == "L2-Bootstrap":
            T_n_Boot = np.zeros((self.N_boot, n_tests))

            if self.Hypothesis == "FAMILY":
                self.hypothesis_LABEL = pair_vec[0]

                yy = np.array([])

                for j in range(self.k_groups):
                    yy = np.append(yy, self.data[j].T)

            _, _, T_n_Boot[:,0] = self.L2Bootstrap(self, yy)

            self.update_family_table(i, [self.N_boot])

        elif self.Hypothesis == "PAIRWISE":
            # n_tests is the number of tests; n = self.N is real n
            for j in range(n_tests):

                self.hypothesis_LABEL = pair_vec[j]
                T = self.setUpTimeBar(i)

                Ct = C[j, :]

                d_points = self.n_domain_points
                k_groups = self.k_groups
                n_iii = self.n_i
                g_data = self.data
                n = self.N

                def _group_booter(g_data, d_points, k_groups, n_iii, n):

                    eta_i_star = np.zeros((d_points, k_groups))
                    build_covar_star = np.zeros((d_points, 0))

                    for k in range(k_groups):
                        indicies = np.random.choice(g_data[k].shape[1], n_iii[k], replace=True)
                        data_subset_boot = g_data[k][:, indicies]

                        eta_i_star[:, k] = np.mean(data_subset_boot, axis=1)

                        zero_mean_data_k_subset = data_subset_boot - eta_i_star[:, k].reshape(-1,1)
                        build_covar_star = np.hstack((build_covar_star, zero_mean_data_k_subset))

                    gamma_hat_star = (1 / (n - k_groups)) * (build_covar_star.T @ build_covar_star)
                    eta_grand_start = np.sum(eta_i_star * n_iii, axis=1) / n

                    return eta_i_star, eta_grand_start, gamma_hat_star

                def _boot_stat():
                    eta_i_star, _, _ = _group_booter(g_data, d_points, k_group, n_iii, n)
                    SSH_t  = ((Ct @ (eta_i_star - eta_i).T)**2) * inv(Ct @ D @ Ct.T)
                    return np.sum(SSH_t)

                with ProcessPoolExecutor() as exe:
                    futures = {exe.submit(_boot_stat): m for m in range(self.N_boot)}
                    for f in as_completed(futures):
                        m = futures[f]
                        T_n_Boot[m, j] = f.result()
                        T.progress()

                    T.stop()
                    T.delete()

            p_value = np.zeros((n_tests, 1))
            for j in range(n_tests):

                p_value[j] = 1 - np.sum(F_n_Boot[:, j] < F_n[j]) / float(self.N_boot)

                critVals[j] = np.quantile(F_n_Boot[:,j], 1 - self.alpha)

                ReversedT_n[j] = np.quantile((critVals[j] / ratio) * f_n_Denominator_Boot[:, j], 1 - self.alpha)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            self.CriticalValues[counter-1][0] = i
            self.CriticalValues[counter-1][1] = ReversedT_n[j]
            self.CriticalValues[counter-1][2] = critVals[j]

        # elif method == "F-Simul":
        #
        #     ratio = (self.N - self.k_groups) / q
        #     T_null = chi_sq_mixture(q, eig_gamma_hat, self.N_simul)
        #     F_null_denom = chi_sq_mixture(self.N - self.k_groups, eig_gamma_hat, self.N_simul)
        #     F_null = (T_null / F_null_denom) * ratio
        #     F_NullFitted = stats.gaussian_kde(F_null)
        #
        #     p_value = np.zeros((n_tests, 1))
        #
        #     for j in range(n_tests):
        #         p_value[j] = 1 - F_NullFitted.integrate_box_1d(-np.inf, F_n[j])
        #         if self.showSimulPlot:
        #             plot_test_stats(p_value[j], self.alpha, F_null, F_n[j], method + " test", self.Hypothesis, pair_vec[j])
        #
        #     pvalue_matrix[:, counter-1] = p_value.flatten()
        #
        #     if self.Hypothesis == "FAMILY":
        #         self.update_family_table(method, [F_NullFitted])
        #
        #     self.CriticalValues[counter-1][0] = method
        #     self.CriticalValues[counter-1][2] = np.quantile(F_null, 1 - self.alpha)
        #     self.CriticalValues[counter-1][1] = np.quantile((self.CriticalValues[counter-1][2] / ratio) * F_null_denom, 1 - self.alpha)
    return pvalue_matrix



def plot_test_stats(p_value, alpha, null_dist, test_stat, test_name, hypothesis, hypothesis_label):

    if p_value <= alpha:
        line_label = f'{test_name} Statistic P-Value: p={p_value:.4f} <= {alpha:.3f}'
        verdict_label = 'Verdict: Reject H_0'
    else:
        line_label = f'{test_name} Statistic P-Value: p={p_value:.4f} > {alpha:.3f}'
        verdict_label = 'Verdict: Fail to Reject H_0'

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.hist(null_dist, bins=100, density=True, alpha=0.7, label='Null Distribution')

    ax.axvline(x=test_stat, color='r', linestyle='--',
               label=f'{line_label}')

    crit_value = np.quantile(null_dist, 1 - alpha)
    if 'F' in test_name:
        null_dist_label = 'Simulated F-type Mixture Null Distribution'
        if hypothesis == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: F-type test'
        else:
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: F-type test'
    else:
        null_dist_label = 'Simulated \chi^2_{1}-type Mixture Null Distribution'
        if hypothesis == "FAMILY":
            title_label = 'One-Way, Family, Functional ANOVA: Squared L-2 Norm test'
        else:
            title_label = f'One-Way, Pairwise ({hypothesis_label}), Functional ANOVA: Squared L-2 Norm test'

    if test_stat > max(null_dist) * 1e2 or test_stat < min(null_dist) * 1e2:
        ax.set_xscale('log')

    x = np.linspace(crit_value, max(null_dist) * 1.1, 100)
    for i in range(len(x)-1):
        ax.fill_between([x[i], x[i+1]], 0, ax.get_ylim()[1], color='red', alpha=0.1)

    ax.axvline(x=crit_value, color='k', linestyle='-',
               label='Beginning of Critical Value Region')

    ax.set_title(f'{title_label}\n{verdict_label}')
    ax.set_xlabel('Null Distribution')
    ax.set_ylabel('PDF')
    ax.legend(loc='best')
    plt.tight_layout()

    plt.show()




# @pytest.fixture(scope="session")
# def dataset():
#     rng = np.random.default_rng(42)
#     d_points = 4
#     n_i = np.array([3, 5])
#     k_groups = len(n_i)
#     g_data = [rng.standard_normal(size=(d_points, n_i[k])) for k in range(k_groups)]
#     return dict(
#         d_points=d_points,
#         k_groups=k_groups,
#         n_i=n_i,
#         n=n_i.sum(),
#         g_data=g_data
#     )
#
#
# def test_group_booter():
#     d = dataset()
#     eta_i_star, eta_grand_star, gamma_star = _group_booter(d["g_data"], d["d_points"], d["k_groups"], d["n_i"], d["n"])
#     assert eta_i_star.shape   == (d["d_points"], d["k_groups"])
#     assert eta_grand_star.shape == (d["d_points"],)
#     assert gamma_star.shape   == (d["n"], d["n"])
#
#     expected = (eta_i_star * d["n_i"]).sum(axis=1) / d["n"]
#     assert np.allclose(eta_grand_star, expected)
#
#     assert np.allclose(gamma_star, gamma_star.T, atol=1e-12)
#
#
# def test_processpool():
#     d = dataset()
#     Ct = np.array([1, -1])
#
#     def boot_stat():
#         eta_i_star, _, _ = _group_booter(
#             d["g_data"], d["d_points"], d["k_groups"], d["n_i"], d["n"]
#         )
#         return float((Ct @ eta_i_star) @ (Ct @ eta_i_star))
#
#     B = 32
#     with ProcessPoolExecutor() as exe:
#         results = list(exe.map(lambda _: boot_stat(), range(B)))
#
#     results = np.asarray(results)
#     assert results.shape == (B,)
#     assert np.all(np.isfinite(results))
#     assert np.all(results >= 0.0)



def _get_group_booter():
    for i in one_way.__code__.co_consts:
        if isinstance(i, types.CodeType) and i.co_name == "_group_booter":
            return types.FunctionType(i, globals(), name="_group_booter")


def dataset():
    rng = np.random.default_rng(42)
    d_points = 4
    n_i = np.array([3, 5])
    k_groups = len(n_i)
    g_data = [rng.standard_normal(size=(d_points, n_i[k])) for k in range(k_groups)]
    return dict(
        d_points=d_points,
        k_groups=k_groups,
        n_i=n_i,
        n=n_i.sum(),
        g_data=g_data
    )


def _single_boot_stat(g_data, d_points, k_groups, n_i, n, Ct):
    boot = _get_group_booter()
    eta_i_star, _, _ = boot(g_data, d_points, k_groups, n_i, n)

    v = Ct @ eta_i_star.T           # shape (d_points,)
    return float(v @ v)             # scalar ≥ 0

def run_group_booter(verbose=True):
    d   = dataset()
    boot = _get_group_booter()
    eta_i_star, eta_grand_star, gamma_star = boot(
        d["g_data"], d["d_points"], d["k_groups"], d["n_i"], d["n"]
    )

    assert eta_i_star.shape   == (d["d_points"], d["k_groups"])
    assert eta_grand_star.shape == (d["d_points"],)
    assert gamma_star.shape   == (d["n"], d["n"])
    assert np.allclose(eta_grand_star,
                       (eta_i_star * d["n_i"]).sum(axis=1) / d["n"])
    assert np.allclose(gamma_star, gamma_star.T, atol=1e-12)

    if verbose:                             # <─ NEW
        print("eta_i_star:\n", eta_i_star)
        print("eta_grand_star:", eta_grand_star)
        print("gamma_star shape:", gamma_star.shape)


def run_processpool(verbose=True):
    d  = dataset()
    Ct = np.array([1, -1], dtype=float)
    B  = 32

    with ProcessPoolExecutor() as exe:
        futures  = [exe.submit(_single_boot_stat,
                               d["g_data"], d["d_points"], d["k_groups"],
                               d["n_i"], d["n"], Ct) for _ in range(B)]
        results  = np.asarray([f.result() for f in futures])

    assert results.shape == (B,)
    assert np.all(np.isfinite(results)) and np.all(results >= 0.0)

    if verbose:                             # <─ NEW
        print("bootstrap statistics:", results)
        print("mean stat =", results.mean(),
              "   max stat =", results.max())

if __name__ == "__main__":
    run_group_booter()
    run_processpool()
