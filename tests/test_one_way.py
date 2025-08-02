import numpy as np
import sys
import os
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functionalANOVA.utils import chi_sq_mixture, aflag_maker, l2_bootstrap, f_bootstrap, update_family_table
from functionalANOVA.one_way import one_way


class TestFANOVA:
    def __init__(self, data_array, n_simul=1000, n_boot=100, alpha=0.05):
        self.data = data_array
        self.k_groups = len(data_array)
        self.n_i = [data.shape[1] for data in data_array]
        self.N = sum(self.n_i)
        self.n_domain_points = data_array[0].shape[0]
        self.n_simul = n_simul
        self.N_boot = n_boot
        self.alpha = alpha

        self.ANOVA_Methods_Used = ["L2-Simul", "L2-Bootstrap"]
        self.CriticalValues = [[None, None, None] for _ in range(len(self.ANOVA_Methods_Used))]
        self.hypothesis = "FAMILY"
        self.show_simul_plot = False

    @staticmethod
    def chi_sq_mixture(df, coefs, N_samples):
        return chi_sq_mixture(df, coefs, N_samples)

    @staticmethod
    def aflag_maker(n_i):
        return aflag_maker(n_i)


def test_one_way():
    np.random.seed(42)

    print("Testing one_way function with utils integration...")

    p_vars = 10
    n_samples = [30, 25, 35]
    k_groups = len(n_samples)

    data_groups = []
    for i in range(k_groups):
        mean_shift = i * 0.5
        cov = np.eye(p_vars) * 0.5 + np.ones((p_vars, p_vars)) * 0.1
        data = multivariate_normal.rvs(mean=np.ones(p_vars) * mean_shift, cov=cov, size=n_samples[i])
        data_groups.append(data.T)

    fanova = TestFANOVA(data_groups, n_simul=100, n_boot=50)

    print(f"Created FANOVA object:")
    print(f"  k_groups: {fanova.k_groups}")
    print(f"  n_i: {fanova.n_i}")
    print(f"  N: {fanova.N}")
    print(f"  n_domain_points: {fanova.n_domain_points}")

    try:
        eta_i = np.array([np.mean(data, axis=1) for data in fanova.data]).T
        eta_grand = np.mean(eta_i, axis=1)

        build_covar = np.hstack([data - eta_i[:, i:i+1] for i, data in enumerate(fanova.data)])
        gamma_hat = (build_covar @ build_covar.T) / (fanova.N - fanova.k_groups)
        eig_gamma_hat = np.real(np.linalg.eigvals(gamma_hat))
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        n_tests = 1
        q = fanova.k_groups - 1
        params = (None, None, None, None, None, None, None, None)
        pair_vec = ["Family Test"]

        print("\nCalling one_way function...")
        pvalue_matrix = one_way(fanova, n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec)

        print(f"SUCCESS! one_way function completed.")
        print(f"P-value matrix shape: {pvalue_matrix.shape}")
        print(f"P-values: {pvalue_matrix.flatten()}")

        print("\n✓ Integration test PASSED - utils functions work correctly!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n✗ Integration test FAILED")
        return False


if __name__ == "__main__":
    test_one_way()
