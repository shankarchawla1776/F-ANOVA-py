from functionalANOVA.core import FunctionalANOVA
import numpy as np
from scipy.stats import multivariate_normal

# Create proper functional data for testing
np.random.seed(42)
p_vars = 10
n_samples = [30, 25, 35]
k_groups = len(n_samples)

data_groups = []
for i in range(k_groups):
    mean_shift = i * 0.5
    cov = np.eye(p_vars) * 0.5 + np.ones((p_vars, p_vars)) * 0.1
    data = multivariate_normal.rvs(mean=np.ones(p_vars) * mean_shift, cov=cov, size=n_samples[i])
    data_groups.append(data.T)

# Initialize FunctionalANOVA with required attributes
fa = FunctionalANOVA(
    data=data_groups,
    k_groups=k_groups,
    n_i=n_samples,
    N=sum(n_samples),
    n_domain_points=p_vars,
    n_simul=100,
    n_boot=50,
    alpha=0.05,
    ANOVA_Methods_Used=["L2-Simul", "L2-Bootstrap"],
    hypothesis="FAMILY",
    show_simul_plot=False
)

# Calculate required parameters
eta_i = np.array([np.mean(data, axis=1) for data in fa.data]).T
eta_grand = np.mean(eta_i, axis=1)

build_covar = np.hstack([data - eta_i[:, i:i+1] for i, data in enumerate(fa.data)])
gamma_hat = (build_covar @ build_covar.T) / (fa.N - fa.k_groups)
eig_gamma_hat = np.real(np.linalg.eigvals(gamma_hat))
eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

# Calculate test statistics
n_tests = 1
q = fa.k_groups - 1

# Calculate SSH (Sum of Squares Hypothesis) for family test
SSH = np.sum(fa.n_i * (eta_i - eta_grand.reshape(-1, 1))**2, axis=1)
T_n = np.array([np.sum(SSH)])  # L2 test statistic
F_n = np.array([T_n[0] / (np.trace(gamma_hat) * q)])  # F test statistic

# Calculate scaling factors
beta_hat = np.trace(gamma_hat) / fa.n_domain_points
kappa_hat = fa.n_domain_points / np.trace(gamma_hat)
beta_hat_unbias = beta_hat
kappa_hat_unbias = kappa_hat

# For family test, we don't need C and D matrices (used for pairwise)
C = None
D = None

params = (T_n, F_n, beta_hat, kappa_hat, beta_hat_unbias, kappa_hat_unbias, C, D)
pair_vec = ["Family Test"]

# Now call one_way_anova with proper parameters
# res = fa.one_way_anova(n_tests, q, eig_gamma_hat, eta_i, eta_grand, params, pair_vec)
res = fa.one_way_anova(n_tests=n_tests,q=q, eig_gamma_hat=eig_gamma_hat, eta_i=eta_i, eta_grand=eta_grand, params=params, pair_vec=pair_vec)
print(f"P-values: {res.flatten()}")
