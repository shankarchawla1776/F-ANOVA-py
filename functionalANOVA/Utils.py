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


def chi_sq_mixture(df, coefs, N_samples):
    n_eigs = len(coefs)
    chi2rvs = np.random.chisquare(df, size=(n_eigs, N_samples))
    T_null = (coefs @ chi2rvs)
    return T_null

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

    v = Ct @ eta_i_star.T
    return float(v @ v)

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

    if verbose:
        print("eta_i_star:", eta_i_star)
        print("eta_grand_star:", eta_grand_star)
        print("gamma_star shape:", gamma_star.shape)


def run_processpool(verbose=True):
    d  = dataset()
    Ct = np.array([1, -1], dtype=float)
    B  = 52

    with ProcessPoolExecutor() as exe:
        futures  = [exe.submit(_single_boot_stat,
                               d["g_data"], d["d_points"], d["k_groups"],
                               d["n_i"], d["n"], Ct) for _ in range(B)]
        results  = np.asarray([f.result() for f in futures])

    assert results.shape == (B,)
    assert np.all(np.isfinite(results)) and np.all(results >= 0.0)

    if verbose:
        print("bootstrap statistics:", results)
        print("mean=", results.mean(),
              "max=", results.max())
