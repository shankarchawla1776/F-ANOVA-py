import numpy as np
import pandas as pd
import types
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from scipy import stats
from scipy.stats import chi2, ncx2, f
from scipy.linalg import inv, sqrtm


def chi_sq_mixture(df, coefs, N_samples):
    n_eigs = len(coefs)
    chi2rvs = np.random.chisquare(df, size=(n_eigs, N_samples))
    T_null = (coefs @ chi2rvs)
    return T_null

def aflag_maker(n_i):
    aflag = []
    for k in range(len(n_i)):
        # indicator = np.repeat(k, n_i[k]) #MATLAB indexing uses 1 start
        indicator = np.repeat(k + 1, n_i[k]) #MATLAB indexing uses 1 start
        aflag.extend(indicator)
    return np.array(aflag)

def l2_bootstrap(obj, yy, method):
    n, p = yy.shape

    gsize = np.array(obj.n_i)
    aflag = aflag_maker(gsize)
    aflag0 = np.unique(aflag)
    k = len(aflag0)

    mu0 = np.mean(yy, axis=0)
    gsize_list = []
    vmu = []
    z = []
    SSR = 0

    for i in range(k):
        iflag = (aflag == aflag0[i])
        yi = yy[iflag, :]

        ni = yi.shape[0]
        mui = np.mean(yi, axis=0)
        zi = yi - np.ones((ni, 1)) @ mui.reshape(1, -1)

        gsize_list.append(ni)
        vmu.append(mui)

        z.append(zi)
        SSR += ni * (mui - mu0)**2

    gsize = np.array(gsize_list)
    vmu = np.array(vmu)

    stat = np.sum(SSR)

    btstat = np.zeros(obj.n_boot)

    for ii in tqdm(range(obj.n_boot), desc=obj._setup_time_bar(method)):
        btvmu = []

        for i in range(k):
            iflag = (aflag == aflag0[i])
            yi = yy[iflag, :]
            ni = gsize[i]

            btflag = np.random.choice(ni, ni, replace=True)
            btyi = yi[btflag, :]

            btmui = np.mean(btyi, axis=0) - vmu[i, :]
            btvmu.append(btmui)

        btvmu = np.array(btvmu)
        btmu0 = (gsize @ btvmu) / n
        btSSR = 0

        for i in range(k):
            btSSR += gsize[i] * (btvmu[i, :] - btmu0)**2

        btstat[ii] = np.sum(btSSR)

    # p_value = np.mean(btstat >= stat)

    return  np.expand_dims(btstat, axis=1)

def f_bootstrap(obj, yy, method):
    n, p = yy.shape

    gsize = np.array(obj.n_i)
    aflag = aflag_maker(gsize)

    aflag0 = np.unique(aflag)
    k = len(aflag0)

    mu0 = np.mean(yy, axis=0)
    gsize_list = []
    vmu = []
    z = []
    SSR = 0

    for i in range(k):
        iflag = (aflag == aflag0[i])

        yi = yy[iflag, :]
        ni = yi.shape[0]

        mui = np.mean(yi, axis=0)
        zi = yi - np.ones((ni, 1)) @ mui.reshape(1, -1)

        gsize_list.append(ni)
        vmu.append(mui)

        z.append(zi)
        SSR += ni * (mui - mu0)**2

    gsize = np.array(gsize_list)
    vmu = np.array(vmu)
    z = np.vstack(z)

    if n > p:
        Sigma = (z.T @ z) / (n - k)
    else:
        Sigma = (z @ z.T) / (n - k)

    A = np.trace(Sigma)
    stat = np.sum(SSR) / A / (k - 1)

    btstat = np.zeros(obj.n_boot)

    for ii in tqdm(range(obj.n_boot), desc=obj._setup_time_bar(method)):
        btvmu = []
        btz = []

        for i in range(k):
            iflag = (aflag == aflag0[i])
            yi = yy[iflag, :]
            ni = gsize[i]
            btflag = np.random.choice(ni, ni, replace=True)

            btyi = yi[btflag, :]
            btmui = np.mean(btyi, axis=0)
            btzi = btyi - np.ones((ni, 1)) @ btmui.reshape(1, -1)
            btz.append(btzi)
            btmui = btmui - vmu[i, :]
            btvmu.append(btmui)

        btvmu = np.array(btvmu)
        btz = np.vstack(btz)

        btmu0 = (gsize @ btvmu) / n
        btSSR = 0

        for i in range(k):
            btSSR += gsize[i] * (btvmu[i, :] - btmu0)**2

        if n > p:
            btSigma = (btz.T @ btz) / (n - k)
        else:
            btSigma = (btz @ btz.T) / (n - k)

        btA = np.trace(btSigma)
        btstat[ii] = np.sum(btSSR) / btA / (k - 1)

    # p_value = np.mean(btstat >= stat)

    return  np.expand_dims(btstat, axis=1) 

def generate_two_way_comb(self):
    combinations = []
    for K in range(self.A_groups):
        for KK in range(self.B_groups):
            combination = f"{self.primary_labels[K]}-{self.secondary_labels[KK]}"
            combinations.append(combination)
    return combinations

def construct_pairwise_contrast_matrix(total_groups: int) -> np.ndarray:
    
    """
    Construct all pairwise contrast coefficient rows for total_groups.
    Returns a contrast matrix C of shape (num_pairs, total_groups).
    """
    k = total_groups

    if k == 2:
        return np.hstack([np.eye(k - 1), -np.ones((k - 1, 1))])

    blocks = []
    for mm in range(k - 1):  # mm = 0 to k-2
        block = np.zeros((k - mm - 1, k))  # <-- must be (rows, k), not (rows, k-1)
        for cc in range(k - mm - 1):
            block[cc, mm] = 1
            block[cc, cc + mm + 1] = -1
        blocks.append(block)

    C = np.vstack(blocks)
    return C

def compute_group_means(k_groups, n_domain_points, data, n_i, N):
    # Initialize
    eta_i = np.zeros((n_domain_points, k_groups))
    build_Covar_star = np.empty((n_domain_points, 0))

    # Loop over groups
    for k in range(k_groups):
        group_data = data[k]  # shape: (n_domain_points, n_samples_in_group)
        eta_i[:, k] = np.mean(group_data, axis=1)  # mean across samples for each timepoint
        zero_mean_subset = group_data - eta_i[:, k, np.newaxis]
        build_Covar_star = np.concatenate((build_Covar_star, zero_mean_subset), axis=1)

    # Compute grand mean
    eta_grand = np.sum(eta_i * np.asarray(n_i)[np.newaxis, :], axis=1) / N
    
    return eta_i, eta_grand, build_Covar_star

def beta_hat(COV):
    """
    Compute beta_hat from a covariance matrix.
    Equivalent to trace(COV^2) / trace(COV)
    """
    return np.trace(COV @ COV) / np.trace(COV)

def kappa_hat(COV):
    """
    Compute kappa_hat from a covariance matrix.
    Equivalent to (trace(COV)^2) / trace(COV^2)
    """
    return (np.trace(COV) ** 2) / np.trace(COV @ COV)

def unbiased_estimator_trace_squared(n, k, COV):
    """
    Unbiased estimator for trace(COV)^2
    """
    n_adj = ((n - k) * (n - k + 1)) / ((n - k - 1) * (n - k + 2))
    factor = np.trace(COV) ** 2 - (2 * np.trace(COV @ COV)) / (n - k + 1)
    return n_adj * factor

def unbiased_estimator_trace_cov_squared(n, k, COV):
    """
    Unbiased estimator for trace(COV @ COV)
    """
    n_adj = ((n - k) ** 2) / ((n - k - 1) * (n - k + 2))
    factor = np.trace(COV @ COV) - (np.trace(COV) ** 2) / (n - k)
    return n_adj * factor

def beta_hat_unbias(n, k, COV):
    trace_cov_squared = unbiased_estimator_trace_cov_squared(n, k, COV)
    return trace_cov_squared / np.trace(COV)

def kappa_hat_unbias(n, k, COV):
    return unbiased_estimator_trace_squared(n, k, COV) / unbiased_estimator_trace_cov_squared(n, k, COV)

def group_booter(data_matrix_cell, n_domain_points, k_groups, n_i, n):
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

def update_family_table(df, method, params):
    # Find the rows where the method matches
    mask = df['Family-Wise Method'] == method

    if method in {"L2-Simul", "F-Simul"}:
        df.loc[mask, 'Parameter 1 Name'] = 'KDE: Kernel'
        df.loc[mask, 'Parameter 2 Name'] = 'KDE: BandWidth'
        df.loc[mask, 'Parameter 1 Value'] = 'Gaussian'
        df.loc[mask, 'Parameter 2 Value'] = params[0].factor

    elif method in {"L2-Bootstrap", "F-Bootstrap"}:
        df.loc[mask, 'Parameter 1 Name'] = 'Bootstrap: Resamples'
        df.loc[mask, 'Parameter 2 Name'] = 'Bootstrap: Type'
        df.loc[mask, 'Parameter 1 Value'] = params[0]
        df.loc[mask, 'Parameter 2 Value'] = 'nonparametric'