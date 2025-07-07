import numpy as np
from scipy import stats
from scipy.stats import chi2, f
from scipy.linalg import inv, sqrtm
import pandas as pd




def one_way_bf(self, method, data, contrast, c, indicator_a=None):

    N = self.N
    p = self.n_domain_points


    if indicator_a is None:
        gsize = self.n_i
        aflag = self.__class__.aflag_maker(gsize)

    else:
        aflag = indicator_a


    yy = data

    aflag0 = np.unique(aflag)

    k = len(aflag0)

    vmu = np.array([])
    A = np.array([])
    A2 = np.array([])
    B2 = np.array([])
    gsize = np.zeros(k)
    S_ii = [None] * k



    for i in range(k):
        iflag = (aflag == aflag0[i])
        yi = yy[iflag, :]
        gsize[i] = yi.shape[0]
        ni = int(gsize[i])
        mui = np.mean(yi, axis=0)

        if vmu.size:
            vmu = np.vstack([vmu, mui])
        else:
            vmu = mui.reshape(1,-1)

        ri = yi - np.ones((ni,1)) @ mui.reshape(1,-1)


        if N > p:
            Si = ri.T @ ri / (ni-1) # p by p case, rank: p leq N-1
        else:
            Si = ri @ ri.T / (ni-1) # n_i by n_i case, full rank to n_i-1


        S_ii[i] = Si
        Ai = np.trace(Si)
        A = np.append(A, Ai)


        if method in ["L2-Naive", "L2-BiasReduced", "F-Naive", "F-BiasReduced"]:
            Bi = np.trace(Si @ Si)

            if method in ["L2-Naive", "F-Naive"]:
                A2i = Ai**2
                B2i = Bi

            elif method in ["L2-BiasReduced", "F-BiasReduced"]:
                A2i = (ni - 1) * ni / (ni - 2) / (ni + 1) * (Ai**2 - 2 * Bi / ni)
                B2i = (ni - 1)**2 / ni / (ni + 1) * (Bi - Ai**2 / (ni - 1))

            A2 = np.append(A2, A2i)
            B2 = np.append(B2, B2i)

    D = np.diag(1/gsize)
    H = sqrtm(inv(contrast @ D @ contrast.T))
    stat0 = np.trace(H @ (contrast @ vmu - c) @ (contrast @ vmu-c).T @ H.T)


    if method in ["L2-Naive", "L2-BiasReduced", "F-Naive", "F-BiasReduced"]:
        Dh = np.sqrt(D)
        W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh

        dd = np.diag(W)
        K1 = np.sum(dd *A)

        K2a = np.sum(dd**2 * A2)
        K2b = np.sum(dd**2 * B2)

        AB1 = []
        AB2 = []


        for i in range(k-1):
            ni = int(gsize[i])
            iflag = (aflag == aflag0[i])

            yi = yy[iflag , :]
            ri = yi - np.ones((ni,1)) @ vmu[i,:].reshape(1,-1)

            for j in range(i+1, k):
                nj = int(gsize[j])
                jflag = (aflag == aflag0[j])
                yj = yy[jflag, :]
                rj = yj - np.ones((nj,1)) @ vmu[j,:].reshape(1,-1)

                if N > p:
                    temp = np.trace(ri.T @ ri @ rj.T @ rj) / (ni - 1) / (nj - 1)
                else:
                    temp = np.trace(ri @ rj.T @ rj @ ri.T) / (ni - 1) / (nj - 1)

                K2a += 2 * W[i, i] * W[j, j] * A[i] * A[j]

                AB1.append(A[i] * A[j])

                K2b += 2 * W[i, j]**2 * temp

                AB2.append(temp)

    if method in ["F-Bootstrap", "F-Simul"]:
        Dh = np.sqrt(D)
        # k by k
        W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh

        dd = np.diag(W)
        K1 = np.sum(dd * A)

        f_stat = stat0 / K1

        if self.Hypothesis == "FAMILY":

            b_n = np.sqrt(gsize)
            A_n = np.eye(k) - np.outer(b_n, b_n) / N
            A_n_ii = np.diag(A_n)

            mask = np.ones(k, dtype=bool)

        elif self.Hypothesis == "PAIRWISE":
            # # np.logical_not makes a boolean mask where true means 0 in contrast. replace with np.any
            # mask = np.logical_not(np.abs(contrast.T))

            mask = np.any(contrast != 0, axis=0)
            g_n = gsize[mask]

            N_n = np.sum(g_n)
            k_n = len(g_n)

            b_n = np.sqrt(g_n)
            A_n = np.eye(k_n) - np.outer(b_n, b_n) / N_n
            A_n_ii = np.diag(A_n)

        elif self.Hypothesis in ['INTERACTION', 'PRIMARY', 'SECONDARY']:

            A_n = D**0.5 @ contrast.T @ inv(contrast @ D @ contrast.T) @ contrast @ D**0.5
            A_n_ii = np.diag(A_n)
            mask = np.ones(k, dtype=bool)



    if method in ["L2-Naive", "L2-BiasReduced"]:
        beta = K2b / K1
        df = K2a / K2b

        stat = stat0 / beta
        pvalue = 1 - chi2.cdf(stat, df)
        pstat = [stat0, pvalue]
        params = [beta, df, K1, K2a, 2*K2b]

    elif method in ["F-Naive", "F-BiasReduced"]:
        f_stat = stat0 / K1
        K2c = np.sum((dd / gsize)**2 * B2 / (gsize - 1))

        df1 = K2a / K2b
        df2 = K2a / K2c

        pvalue = 1 - f.cdf(f_stat, df1, df2)
        pstat = [f_stat, pvalue]
        params = [df1, df2, K2a, 2*K2b, 2*K2c]

    elif method == "L2-Bootstrap":
        Bstat = np.zeros(self.N_boot)

        ts = self.setUpTimeBar(method)

        for ii in range(self.N_boot):

            Bmu = np.empty((0, p))
            for i in range(k):
                iflag = (aflag == aflag0[i])
                yi = yy[iflag, :]
                ni = int(gsize[i])

                Bflag = np.random.choice(ni, ni, replace=True)
                Byi = yi[Bflag, :]
                Bmui = np.mean(Byi, axis=0)
                Bmu = np.vstack([Bmu, Bmui])

            temp = H @ contrast @ (Bmu - vmu)
            temp = np.trace(temp @ temp.T)
            Bstat[ii] = temp

            ts.progress()

        ts.stop()
        ts.delete()

        pvalue = np.mean(Bstat > stat0)
        pstat = [stat0, pvalue]

    elif method == "F-Bootstrap":
        Bstat = np.zeros(self.N_boot)
        ts = self.setUpTimeBar(method)

        for ii in range(self.N_boot):
            Bmu = np.empty((0, p))
            tr_gamma = []

            for i in range(k):
                iflag = (aflag == aflag0[i])
                yi = yy[iflag, :]
                ni = int(gsize[i])

                Bflag = np.random.choice(ni, ni, replace=True)
                Byi = yi[Bflag, :]

                Bmui = np.mean(Byi, axis=0)
                Bmu = np.vstack([Bmu, Bmui])

                if mask[i]:
                    # stats for ith group in k
                    z_mean = Byi - Bmui
                    test_cov = (z_mean @ z_mean.T) / (ni - 1)
                    tr_gamma_i = np.trace(test_cov)

                    tr_gamma.append(tr_gamma_i)

            temp = H @ contrast @ (Bmu - vmu)
            T_n = np.trace(temp @ temp.T)

            S_n = np.sum(A_n_ii * tr_gamma)
            temp = T_n / S_n

            Bstat[ii] = temp

            ts.progress()

        ts.stop()
        ts.delete()

        pvalue = np.mean(Bstat > f_stat)
        pstat = [f_stat, pvalue]

    elif method == "L2-Simul":
        if self.Hypothesis in ['FAMILY', 'PAIRWISE']:
            build_covar_star = np.zeros((self.n_domain_points, 0))

            mask = np.any(np.logical_not(contrast.T), axis=1)
            COV_Sum = 0
            vmu = np.empty((0, p))

            for i in range(k):
                if mask[i]:
                    iflag = (aflag == aflag0[i])
                    yi = yy[iflag, :]

                    gsize[i] = yi.shape[0]
                    ni = int(gsize[i])

                    mui = np.mean(yi, axis=0)
                    vmu = np.vstack([vmu, mui])

                    ri = yi - np.ones((ni, 1)) @ mui.reshape(1, -1)
                    COV_Sum += np.cov(ri.T) * (ni - 1)

                    build_covar_star = np.hstack([build_covar_star, ri.T])

            g_n = gsize[mask]
            N_n = np.sum(g_n)
            k_n = len(g_n)

            COV_Sum = COV_Sum / (N_n - k_n)

            eig_gamma_hat = np.linalg.eigvals(COV_Sum)
            eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

            q = k_n - 1
            T_null = self.__class__.chi_sq_mixture(q, eig_gamma_hat, self.N_simul)

            T_NullFitted = stats.gaussian_kde(T_null)
            pvalue = 1 - T_NullFitted.integrate_box_1d(-np.inf, stat0)

            pstat = [stat0, pvalue]
        else:
            pstat = [stat0, np.nan]

    elif method == "F-Simul":
        Dh = np.sqrt(D)
        W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh  # kxk

        dd = np.diag(W)
        K1 = np.sum(dd * A)
        f_stat = stat0 / K1

        if self.Hypothesis in ['FAMILY', 'PAIRWISE']:
            build_covar_star = np.zeros((self.n_domain_points, 0))
            COV_Sum = 0
            vmu = np.empty((0, p))

            for i in range(k):
                if mask[i]:
                    iflag = (aflag == aflag0[i])
                    yi = yy[iflag, :]
                    gsize[i] = yi.shape[0]

                    ni = int(gsize[i])
                    mui = np.mean(yi, axis=0)
                    vmu = np.vstack([vmu, mui])

                    ri = yi - np.ones((ni, 1)) @ mui.reshape(1, -1)
                    COV_Sum += np.cov(ri.T) * (ni - 1)

                    build_covar_star = np.hstack([build_covar_star, ri.T])

            g_n = gsize[mask]
            N_n = np.sum(g_n)
            k_n = len(g_n)

            COV_Sum = COV_Sum / (N_n - k_n)

            eig_gamma_hat = np.linalg.eigvals(COV_Sum)
            eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

            q = k_n - 1
            T_null = self.__class__.chi_sq_mixture(q, eig_gamma_hat, self.N_simul)

            S_null = np.zeros(self.N_simul)
            S_ii_subset = [S_ii[i] for i in range(k) if mask[i]]

            for i in range(k_n):
                eig_gamma_hat = np.linalg.eigvals(S_ii_subset[i])
                eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

                S_temp = self.__class__.chi_sq_mixture(int(g_n[i]) - 1, eig_gamma_hat, self.N_simul)
                S_temp = (S_temp * A_n_ii[i]) / (g_n[i] - 1)
                S_null += S_temp

            F_null = T_null / S_null
            F_NullFitted = stats.gaussian_kde(F_null)

            pvalue = 1 - F_NullFitted.integrate_box_1d(-np.inf, f_stat)

            pstat = [f_stat, pvalue]
        else:
            pstat = [f_stat, np.nan]

    stat = pstat[0]
    pvalue = pstat[1]

    return pvalue, stat


# def aflag_maker(n_i):
#     aflag = []
#     for k in range(len(n_i)):
#         # indicator = np.repeat(k, n_i[k]) #MATLAB indexing uses 1 start
#         indicator = np.repeat(k + 1, n_i[k]) #MATLAB indexing uses 1 start
#         aflag.extend(indicator)
#     return np.array(aflag)
