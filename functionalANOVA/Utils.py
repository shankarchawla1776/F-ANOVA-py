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
