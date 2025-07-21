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

def set_up_two_way(self):
    if hasattr(self, 'subgroup_indicator') and self.subgroup_indicator is not None:
        self.B_groups = len(np.unique(self.subgroup_indicator))
    else:
        self.B_groups = 1

    self.A_groups = self.k_groups
    self.AB_groups = self.A_groups * self.B_groups

    if (not hasattr(self, 'primary_labels') or self.primary_labels is None) and (not hasattr(self, 'secondary_labels') or self.secondary_labels is None):
        self.generic_group_labels = True

    if not hasattr(self, 'primary_labels') or self.primary_labels is None:
        self.primary_labels = [str(i+1) for i in range(len(self.data))]

    if not hasattr(self, 'secondary_labels') or self.secondary_labels is None:
        if hasattr(self, 'subgroup_indicator') and self.subgroup_indicator is not None:
            unique_indicators = np.unique(self.subgroup_indicator)
            self.secondary_labels = [chr(65 + i) for i in range(len(unique_indicators))]
        else:
            self.secondary_labels = ['A']

    assert len(self.primary_labels) == self.A_groups, "Labels for each Primary factor level must have a one-to-one correspondence to each level"

    assert len(self.secondary_labels) == self.B_groups, "Labels for each Secondary factor level must have a one-to-one correspondence to each level"

    if hasattr(self, 'group_labels') and self.group_labels is not None:
        assert self.group_labels is None, 'TwoWay ANOVA requires using "primary_labels" and "secondary_labels" as input arguments.\nIt doesnt support the "group_labels" argument due to ambiguity.'

def generate_two_way_comb(self):
    combinations = []
    for K in range(self.A_groups):
        for KK in range(self.B_groups):
            combination = f"{self.primary_labels[K]}-{self.secondary_labels[KK]}"
            combinations.append(combination)
    return combinations
