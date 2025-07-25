import numpy as np
from scipy.stats import chi2, gaussian_kde
from .utils import chi_sq_mixture, aflag_maker

def k_group_cov(self, method, stat, Sigma, V):
    gsize = self.n_i
    N = self.N
    k = self.k_groups
    p = self.n_domain_points

    if method == "L2-Simul":
        q = self.k_groups - 1
        v_array = []
        for K in range(self.k_groups):
            v_array.append(self.data[K] - np.mean(self.data[K], axis=1, keepdims=True))

        n_array = self.n_i
        LHS = 0

        for ii in range(self.k_groups):
            n_i = n_array[ii]
            V_i = v_array[ii]
            for jj in range(n_i):
                v_ij = V_i[:, jj]
                LHS += np.outer(v_ij, v_ij) * np.outer(v_ij, v_ij)

        LHS /= N

        if LHS.shape == Sigma.shape:
            omega_hat = LHS - (Sigma @ Sigma)
        else:
            vmu = []
            V_large = []

            for ii in range(self.k_groups):
                yyi = self.data[ii].T
                mui = np.mean(yyi, axis=0)

                Vi = yyi - np.ones((self.n_i[ii], 1)) * mui
                vmu.append(mui)
                V_large.append(Vi)

            V_large = np.vstack(V_large)
            SigmaLarge = (V_large.T @ V_large) / (self.N - self.k_groups)
            omega_hat = LHS - (SigmaLarge @ SigmaLarge)

        eig_gamma_hat = np.real(np.linalg.eigvals(omega_hat))
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        T_null = chi_sq_mixture(q, eig_gamma_hat, self.n_simul)
        kde = gaussian_kde(T_null)
        pvalue = 1 - kde.integrate_box_1d(-np.inf, stat)
        pvalue = max(0,min(1,pvalue))

    elif method == "L2-Naive":
        an = np.trace(Sigma)

        # np.lingalg.matrix_power works better than dot products
        bn = np.trace(np.linalg.matrix_power(Sigma, 2))
        cn = np.trace(np.linalg.matrix_power(Sigma, 3))
        dn = np.trace(np.linalg.matrix_power(Sigma, 4))

        Bn = (N - 2)**2 / (N * (N - 3)) * (bn - an**2 / (N - 2))
        Cn = (cn - (3 / (N - 2)) * Bn * an) / (1 + 3 / (N - 2))
        Dn = (dn - (6 / (N - 2)) * Cn * an) / (1 + 6 / (N - 2))

        A_corr = Bn + an**2
        B_corr = 2 * Dn + 2 * Bn**2

        alpha = B_corr / A_corr
        df = (k - 1) * (A_corr**2) / B_corr

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "L2-BiasReduced":
        A0 = np.trace(Sigma @ Sigma) + np.trace(Sigma)**2

        B0 = 2 * np.trace(np.linalg.matrix_power(Sigma, 4)) + 2 * (np.trace(Sigma @ Sigma))**2


        alpha = ((N - k)**2 / (N * (N - k - 1)) * (B0 - A0**2 / (N - k))) / A0
        df = (1 + 1 / (N - k)) * (A0**2 - 2 * B0 / (N - k + 1)) / (B0 - A0**2 / (N - k))
        df = (k - 1) * df

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "Bootstrap-Test":
        aflag = aflag_maker(gsize)
        aflag0 = np.unique(aflag)
        vstat = np.zeros(self.n_boot)

        for ii in range(self.n_boot):
            flag = np.random.choice(N, N, replace=True)
            py = V[flag, :]

            R = []

            for i in range(k):
                iflag = (aflag == aflag0[i])
                yyi = py[iflag, :]
                ni = gsize[i]
                mui = np.mean(yyi, axis=0)
                Ri = yyi - np.ones((ni, 1)) * mui
                R.append(Ri)

            R = np.vstack(R)

            if N > p:
                p_s = (R.T @ R) / (N - k)
            else:
                p_s = (R @ R.T) / (N - k)

            stat0 = 0
            nni = 0
            for i in range(k):
                ni = gsize[i]
                flag = np.arange(nni, nni + ni)
                Ri = R[flag, :]

                if N > p:
                    p_si = (Ri.T @ Ri) / (ni - 1)
                    temp = np.trace(np.linalg.matrix_power(p_si - p_s, 2))
                else:
                    p_si = (Ri @ Ri.T) / (ni - 1)
                    temp = np.trace(np.linalg.matrix_power(p_si, 2)) - 2 * np.trace(((Ri @ R.T) @ R) @ Ri.T) / (N - k) / (ni - 1) + np.trace(np.linalg.matrix_power(p_s, 2))

                stat0 += (ni - 1) * temp
                nni += ni
            vstat[ii] = stat0
        pvalue = 1 - np.sum(vstat < stat) / self.n_boot

    elif method == "Permutation-Test":
        aflag = aflag_maker(gsize)
        aflag0 = np.unique(aflag)

        vstat = np.zeros(self.N_permutations)
        for ii in range(self.N_permutations):
            flag = np.random.permutation(N)
            py = V[flag, :]

            R = []
            for i in range(k):
                iflag = (aflag == aflag0[i])
                yyi = py[iflag, :]
                ni = gsize[i]
                mui = np.mean(yyi, axis=0)
                Ri = yyi - np.ones((ni, 1)) * mui
                R.append(Ri)

            R = np.vstack(R)

            if N > p:
                p_s = (R.T @ R) / (N - k)
            else:
                p_s = (R @ R.T) / (N - k)

            stat0 = 0
            nni = 0
            for i in range(k):
                ni = gsize[i]
                flag = np.arange(nni, nni + ni)
                Ri = R[flag, :]

                if N > p:
                    p_si = (Ri.T @ Ri) / (ni - 1)
                    temp = np.trace(np.linalg.matrix_power(p_si - p_s, 2))

                else:
                    p_si = (Ri @ Ri.T) / (ni - 1)
                    temp = np.trace(np.linalg.matrix_power(p_si, 2)) - 2 * np.trace(((Ri @ R.T) @ R) @ Ri.T) / (N - k) / (ni - 1) + np.trace(np.linalg.matrix_power(p_s, 2))

                stat0 += (ni - 1) * temp
                nni += ni
            vstat[ii] = stat0

        pvalue = 1 - np.sum(vstat < stat) / self.N_permutations

    return pvalue
