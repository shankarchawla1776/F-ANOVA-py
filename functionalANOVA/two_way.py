import numpy as np
from scipy import stats
from scipy.stats import chi2, f
from scipy.linalg import inv
import pandas as pd

from .timer import TimeBar
from .utils import aflag_maker, chi_sq_mixture

def two_way(self, method, data, contrast):

    N = self.N
    ddim = self.n_domain_points
    aflag = aflag_maker(self.n_i)
    bflag = self.SubgroupIndicator
    aflag0 = np.unique(aflag)
    p = len(aflag0)

    yy = data

    bflag0 = np.unique(self.SubgroupIndicator)

    q = len(bflag0)
    k = (p * q)

    gsize = []
    vmu = []
    V = []
    p_cell = []
    error_string = []

    for i in range(p):
        q_cell = []
        for j in range(q):
            ijflag = (aflag == aflag0[i]) & (bflag == bflag0[j])
            ni = np.sum(ijflag)

            if ni == 0:
                error_string.append(f"A missing combination of data occurs for Primary Label {self.PrimaryLabels[i]} and Secondary Label {self.SecondaryLabels[j]}")

            gsize.append(ni)
            yyi = yy[ijflag, :]
            cell_mui = np.mean(yyi, axis=0)
            vmu.append(cell_mui)
            V.extend(yyi - cell_mui)
            q_cell.append(ni)
        p_cell.append(q_cell)

    if error_string:
        for i in range(len(error_string)):
            print(f"warning: {i}")

        raise ValueError("'Missing Combinations Listed Above'")

    self.n_ii = p_cell
    n_ii = np.array([i for j in p_cell for i in j])

    gsize = np.array(gsize)

    vmu = np.array(vmu)
    V = np.array(V)

    if self.Hypothesis not in ["PAIRWISE", "FAMILY"]:
        if self.Weights == "UNIFORM":
            u = np.ones(p) / p
            v = np.ones(q) / q

        elif self.Weights == "PROPORTIONAL":
            matsize = np.array(p_cell)
            N_total = np.sum(gsize)

            u = np.sum(matsize, axis=1) / N_total
            v = np.sum(matsize, axis=0) / N_total

        Hp = np.hstack([np.eye(p-1), - np.ones((p-1,1))])
        Hq = np.hstack([np.eye(q-1), - np.ones((q-1,1))])

        Ap = np.eye(p) - np.outer(np.ones(p), u)
        Aq = np.eye(q) - np.outer(np.ones(q), v)

    if self.Hypothesis in ["PAIRWISE", "FAMILY"]:
        r = contrast.shape[0]
    elif self.Hypothesis == "INTERACTION":
        contrast = np.kron(Hp, Hq) @ np.kron(Ap, Aq)
        r = ((p-1) * (q-1))

    elif self.Hypothesis == "PRIMARY":
        contrast = Hp @ np.kron(Ap, v.reshape(-1,1))
        r = p - 1
    elif self.Hypothesis == "SECONDARY":
        contrast = Hq @ np.kron(u.reshape(1,-1), Aq)
        r = q - 1

    else:
        if self.Contrast_Factor ==1:
            contrast = contrast @ np.kron(Ap, v.reshape(1,-1))
        elif self.Contrast_Factor == 2:
            contrast = contrast @ np.kron(u.reshape(1,-1),Aq)
        r = contrast.shape[0]


    W = inv(contrast @ np.diag(1/gsize) @ contrast.T)
    SSH = np.diag((contrast @ vmu).T @ W @ (contrast @ vmu))
    SSH0 = np.sum(SSH)

    SSE = np.diag(V.T @ V)
    SSE0 = np.sum(SSE)

    pool_coeff = N-k

    if N > ddim:
        Pooled_COVAR = (V.T @ V)/pool_coeff
    else:
        Pooled_COVAR = (V @ V.T)/pool_coeff

    A = np.trace(Pooled_COVAR)
    B = np.trace(Pooled_COVAR @ Pooled_COVAR)

    if "Naive" in method:
        A2 = A**2
        B2 = B

    if "BiasReduced" in method:
        pool_coeff = N-k
        # just used pool_coef instead of N-k explicitly
        A2 = (pool_coeff) * (pool_coeff+1) / (pool_coeff-1) / (pool_coeff+2) * (A**2-2*B/(pool_coeff+1))
        B2 = (pool_coeff)**2 / (pool_coeff-1)/(pool_coeff+2) * (B-A**2/(pool_coeff))

    if method == "L2-Simul":
        stat = SSH0
        eig_gamma_hat = np.linalg.eigvals(Pooled_COVAR)
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        SSH_null = chi_sq_mixture(r, eig_gamma_hat, self.n_simul)
        SSH_NullFitted = stats.gaussian_kde(SSH_null)
        pvalue = 1 - SSH_NullFitted.integrate_box_1d(-np.inf, stat)
        pvalue = max(0,min(1,pvalue))
        pstat = [stat, pvalue]

    elif method in ["L2-Naive", "L2-BiasReduced"]:
        stat = SSH0
        beta = B2/A
        kappa = A2/B2
        pvalue = 1 - chi2.cdf(stat/beta, r * kappa)
        pstat = [stat/beta, pvalue]

    elif method in ["F-Naive", "F-BiasReduced"]:
        stat = SSH0 / SSE0 * (N-k) / r
        kappa = A2/B2
        pvalue = 1 - f.cdf(stat, r * kappa, (N-k) * kappa)
        pstat = [stat, pvalue]

    elif method == "F-Simul":
        stat = SSH0 / SSE0 * (N-k) / r
        eig_gamma_hat = np.linalg.eigvals(Pooled_COVAR)
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        SSH_null = chi_sq_mixture(r, eig_gamma_hat, self.n_simul)
        SSE_null = chi_sq_mixture(N-k, eig_gamma_hat, self.n_simul)

        ratio = (N-k) / r

        F_null = SSH_null / SSE_null * ratio
        F_NullFitted = stats.gaussian_kde(F_null)
        pvalue = 1 - F_NullFitted.integrate_box_1d(-np.inf, stat)
        pvalue = max(0,min(1,pvalue))
        pstat = [stat, pvalue]

    elif method == "L2-Bootstrap":
        stat = SSH0
        Bstat = np.zeros(self.n_boot)


        # ts = self.set_up_time_bar(method)
        ts = TimeBar(self.n_boot, method)

        for b in range(self.n_boot):
            Bmu = []
            V_boot = []
            counter = 0

            for i in range(p):
                for j in range(q):
                    ijflag = (aflag==aflag0[i]) & (bflag == bflag0[j])
                    ni = n_ii[counter]
                    yi = yy[ijflag,:]
                    Bootflag = np.random.choice(ni,ni,replace=True)

                    Byi = yi[Bootflag,:]
                    Bmui = np.mean(Byi, axis=0)
                    Bmu.append(Bmui)

                    counter += 1

            Bmu = np.array(Bmu)
            W = inv(contrast @ np.diag(1/gsize) @ contrast.T)
            SSH = np.diag(((Bmu - vmu) @ contrast.T).T @ W @ (contrast @ (Bmu - vmu).T))
            Bstat[b] = np.sum(SSH)
            ts.progress()

        ts.stop()
        ts.delete()

        pvalue = np.mean(Bstat>stat)
        pstat = [stat, pvalue]


    elif method == "F-Bootstrap":
        stat = SSH0 / SSE0 * (N-k) / r
        # ts = self.set_up_time_bar(method)
        ts = TimeBar(self.n_boot, method)
        ratio = (N-k)/r
        Bstat = np.zeros(self.n_boot)

        for b in range(self.n_boot):
            Bmu = []
            V = []
            counter = 0

            for i in range(p):
                for j in range(q):
                    ijflag = (aflag == aflag0[i]) & (bflag == bflag0[j])
                    ni = n_ii[counter]

                    yi = yy[ijflag,:]

                    Bootflag=np.random.choice(ni, ni, replace=True)
                    Byi = yi[Bootflag,:]
                    Bmui = np.mean(Byi, axis=0)
                    Bmu.append(Bmui)

                    V.extend(Byi - Bmui)

                    counter += 1

            Bmu = np.array(Bmu)
            V = np.array(V)

            W = inv(contrast @ np.diag(1/gsize) @ contrast.T)
            SSH = np.diag(((Bmu - vmu) @ contrast.T).T @ W @ (contrast @ (Bmu-vmu).T))
            SSE = np.diag(V.T @ V)

            Bstat[b] = np.sum(SSH) / np.sum(SSE) * ratio
            ts.progress()

        ts.stop()
        ts.delete()
        pvalue = np.mean(Bstat > stat)
        pstat = [stat, pvalue]

    if len(pstat) > 1:
        pvalue = pstat[1]
    else:
        pvalue = np.nan

    stat = pstat[0]

    return pvalue, stat
