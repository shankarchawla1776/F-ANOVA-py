import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functionalANOVA.two_group_cov import two_group_cov


class MockFANOVA:
    def __init__(self, n_simul=1000, n_boot=100, n_permutations=100):
        self.N_simul = n_simul
        self.N_boot = n_boot
        self.N_permutations = n_permutations
    
    @staticmethod
    def chi_sq_mixture(df, coefs, N_samples):
        import numpy as np
        n_eigs = len(coefs)
        chi2rvs = np.random.chisquare(df, size=(n_eigs, N_samples))
        T_null = (coefs @ chi2rvs)
        return T_null


@pytest.mark.homo
def test_homoscedastic_covariance():
    np.random.seed(42)
    
    n1, n2 = 50, 50
    p = 10
    
    true_cov = np.eye(p) * 0.5 + np.ones((p, p)) * 0.1
    
    y1 = multivariate_normal.rvs(mean=np.zeros(p), cov=true_cov, size=n1)
    y2 = multivariate_normal.rvs(mean=np.zeros(p), cov=true_cov, size=n2)
    
    fanova = MockFANOVA(n_simul=1000, n_boot=100, n_permutations=100)
    
    methods = ["L2-Simul", "L2-Naive", "L2-BiasReduced", "Bootstrap-Test", "Permutation-Test"]
    pvalues = {}
    
    for method in methods:
        pvalue = two_group_cov(fanova, method, y1, y2)
        pvalues[method] = pvalue
        print(f"{method}: p-value = {pvalue:.4f}")
    
    print(f"\nhomoscedastic test p-values:")
    for method, pvalue in pvalues.items():
        print(f"  {method}: {pvalue:.6f}")
    
    # Calculate covariance difference
    cov1 = np.cov(y1.T)
    cov2 = np.cov(y2.T)
    cov_diff = cov1 - cov2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('homoscedastic data - two group covariance test', fontsize=16)
    
    axes[0, 0].scatter(y1[:, 0], y1[:, 1], alpha=0.6, label='group 1', color='blue')
    axes[0, 0].scatter(y2[:, 0], y2[:, 1], alpha=0.6, label='group 2', color='red')
    axes[0, 0].set_xlabel('variable 1')
    axes[0, 0].set_ylabel('variable 2')
    axes[0, 0].set_title('scatter plot of first two variables')
    axes[0, 0].legend()
    
    im1 = axes[0, 1].imshow(cov1, cmap='viridis')
    axes[0, 1].set_title('group 1 covariance matrix')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(cov2, cmap='viridis')
    axes[0, 2].set_title('group 2 covariance matrix')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(cov_diff, cmap='RdBu', vmin=-np.abs(cov_diff).max(), vmax=np.abs(cov_diff).max())
    axes[1, 0].set_title('covariance difference (group1 - group2)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Add density plots
    from scipy.stats import gaussian_kde
    data_combined = np.vstack([y1[:, 0], y2[:, 0]])
    kde1 = gaussian_kde(y1[:, 0])
    kde2 = gaussian_kde(y2[:, 0])
    x_range = np.linspace(data_combined.min(), data_combined.max(), 200)
    
    axes[1, 1].plot(x_range, kde1(x_range), label='group 1 density', color='blue')
    axes[1, 1].plot(x_range, kde2(x_range), label='group 2 density', color='red')
    axes[1, 1].set_xlabel('variable 1 values')
    axes[1, 1].set_ylabel('density')
    axes[1, 1].set_title('density estimates for variable 1')
    axes[1, 1].legend()
    
    methods_short = [m.replace('-', '\n') for m in methods]
    bars = axes[1, 2].bar(methods_short, list(pvalues.values()))
    axes[1, 2].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[1, 2].set_ylabel('p-value')
    axes[1, 2].set_title('p-values by method')
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/shankarchawla/math/projects/fanova/F-ANOVA-py/tests/homoscedastic_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    high_pvalue_methods = sum(1 for p in pvalues.values() if p > 0.05)
    
    print(f"\nhomoscedastic test results:")
    print(f"methods with p-value > 0.05: {high_pvalue_methods}/{len(methods)}")
    print(f"expected: most methods should have high p-values")
    
    assert high_pvalue_methods >= 3, f"expected most methods to have high p-values, got {high_pvalue_methods}/{len(methods)}"


@pytest.mark.hetero
def test_heteroscedastic_covariance():
    np.random.seed(123)
    
    n1, n2 = 50, 50
    p = 10
    
    cov1 = np.eye(p) * 0.2 + np.ones((p, p)) * 0.05
    cov2 = np.eye(p) * 2.0 + np.ones((p, p)) * 0.5
    
    y1 = multivariate_normal.rvs(mean=np.zeros(p), cov=cov1, size=n1)
    y2 = multivariate_normal.rvs(mean=np.zeros(p), cov=cov2, size=n2)
    
    fanova = MockFANOVA(n_simul=1000, n_boot=100, n_permutations=100)
    
    methods = ["L2-Simul", "L2-Naive", "L2-BiasReduced", "Bootstrap-Test", "Permutation-Test"]
    pvalues = {}
    
    for method in methods:
        pvalue = two_group_cov(fanova, method, y1, y2)
        pvalues[method] = pvalue
        print(f"{method}: p-value = {pvalue:.4f}")
    
    print(f"\nheteroscedastic test p-values:")
    for method, pvalue in pvalues.items():
        print(f"  {method}: {pvalue:.6f}")
    
    # Calculate covariance difference
    cov1 = np.cov(y1.T)
    cov2 = np.cov(y2.T)
    cov_diff = cov1 - cov2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('heteroscedastic data - two group covariance test', fontsize=16)
    
    axes[0, 0].scatter(y1[:, 0], y1[:, 1], alpha=0.6, label='group 1 (low var)', color='blue')
    axes[0, 0].scatter(y2[:, 0], y2[:, 1], alpha=0.6, label='group 2 (high var)', color='red')
    axes[0, 0].set_xlabel('variable 1')
    axes[0, 0].set_ylabel('variable 2')
    axes[0, 0].set_title('scatter plot of first two variables')
    axes[0, 0].legend()
    
    im1 = axes[0, 1].imshow(cov1, cmap='viridis')
    axes[0, 1].set_title('group 1 covariance matrix (low var)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(cov2, cmap='viridis')
    axes[0, 2].set_title('group 2 covariance matrix (high var)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[1, 0].imshow(cov_diff, cmap='RdBu', vmin=-np.abs(cov_diff).max(), vmax=np.abs(cov_diff).max())
    axes[1, 0].set_title('covariance difference (group1 - group2)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Add density plots
    from scipy.stats import gaussian_kde
    data_combined = np.vstack([y1[:, 0], y2[:, 0]])
    kde1 = gaussian_kde(y1[:, 0])
    kde2 = gaussian_kde(y2[:, 0])
    x_range = np.linspace(data_combined.min(), data_combined.max(), 200)
    
    axes[1, 1].plot(x_range, kde1(x_range), label='group 1 density', color='blue')
    axes[1, 1].plot(x_range, kde2(x_range), label='group 2 density', color='red')
    axes[1, 1].set_xlabel('variable 1 values')
    axes[1, 1].set_ylabel('density')
    axes[1, 1].set_title('density estimates for variable 1')
    axes[1, 1].legend()
    
    methods_short = [m.replace('-', '\n') for m in methods]
    bars = axes[1, 2].bar(methods_short, list(pvalues.values()))
    axes[1, 2].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[1, 2].set_ylabel('p-value')
    axes[1, 2].set_title('p-values by method')
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/shankarchawla/math/projects/fanova/F-ANOVA-py/tests/heteroscedastic_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    low_pvalue_methods = sum(1 for p in pvalues.values() if p < 0.05)
    
    print(f"\nheteroscedastic test results:")
    print(f"methods with p-value < 0.05: {low_pvalue_methods}/{len(methods)}")
    print(f"expected: most methods should have low p-values")
    
    assert low_pvalue_methods >= 3, f"expected most methods to have low p-values, got {low_pvalue_methods}/{len(methods)}"


if __name__ == "__main__":
    print("running homoscedastic test...")
    test_homoscedastic_covariance()
    
    print("\nrunning heteroscedastic test...")
    test_heteroscedastic_covariance()
    
    print("\nboth tests completed successfully!")