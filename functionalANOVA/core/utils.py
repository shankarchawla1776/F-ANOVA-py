import numpy as np
import pandas as pd
import types
import matplotlib.pyplot as plt
import matplotlib

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

def l2_bootstrap(self, yy):
    n, p = yy.shape

    gsize = np.array(self.n_i)
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

    btstat = np.zeros(self.n_boot)

    for ii in range(self.n_boot):
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

    p_value = np.mean(btstat >= stat)

    return p_value, stat, btstat

def f_bootstrap(self, yy):
    n, p = yy.shape

    gsize = np.array(self.n_i)
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

    btstat = np.zeros(self.n_boot)

    for ii in range(self.n_boot):
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

    p_value = np.mean(btstat >= stat)

    return p_value, stat, btstat

def update_family_table(self, A_method, params):
    if A_method in ["L2-Simul", "F-Simul"]:

        if hasattr(self, 'OneWay_P_Table') and self.OneWay_P_Table is not None:
            mask = self.OneWay_P_Table.iloc[:, 0] == A_method
            self.OneWay_P_Table.loc[mask, "Parameter 1 Name"] = 'KDE: Kernel'
            self.OneWay_P_Table.loc[mask, "Parameter 2 Name"] = 'KDE: BandWidth'

            if hasattr(params[0], 'Kernel'):
                self.OneWay_P_Table.loc[mask, "Parameter 1 Value"] = params[0].Kernel

            if hasattr(params[0], 'Bandwidth'):
                self.OneWay_P_Table.loc[mask, "Parameter 2 Value"] = params[0].Bandwidth

    elif A_method in ["L2-Bootstrap", "F-Bootstrap"]:
        if hasattr(self, 'OneWay_P_Table') and self.OneWay_P_Table is not None:
            mask = self.OneWay_P_Table.iloc[:, 0] == A_method
            self.OneWay_P_Table.loc[mask, "Parameter 1 Name"] = 'Bootstrap: Resamples'
            self.OneWay_P_Table.loc[mask, "Parameter 2 Name"] = 'Bootstrap: Type'
            self.OneWay_P_Table.loc[mask, "Parameter 1 Value"] = params[0]
            self.OneWay_P_Table.loc[mask, "Parameter 2 Value"] = "nonparametric"


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
        block = np.zeros((k - mm - 1, k - 1))
        for cc in range(k - mm - 1):
            block[cc, mm] = 1
            block[cc, cc + mm + 1] = -1
        blocks.append(block)

    C = np.vstack(blocks)
    return C