import numpy as np
from scipy.stats import chi2, gaussian_kde
from functionalANOVA.core import utils

# TODO: Needs work
def k_group_cov_pairwise(self, method, y1, y2):

    n1, L = y1.shape
    n2, _ = y2.shape
    gsize = [n1, n2]

    Sigma1 = np.cov(y1, rowvar=False)
    Sigma2 = np.cov(y2, rowvar=False)
    N = n1 + n2
    # Sigma = ((n1-1)*Sigma1 + (n2-1)*Sigma2) / (N-2)
    stat = (n1 - 1) * (n2 - 1) / (N - 2) * np.trace((Sigma1 - Sigma2) @ (Sigma1 - Sigma2))  # Corrected

    # TESTING
    vmu = []
    V = []
    n_i = [n1, n2]
    temp_data = [y1, y2]

    for ii in range(2):
        yyi = temp_data[ii]
        mui = np.mean(yyi, axis=0)
        Vi = yyi - np.ones((n_i[ii], 1)) @ mui.reshape(1, -1)
        vmu.append(mui)
        V.append(Vi)

    vmu = np.array(vmu)
    V = np.vstack(V)

    m = self.n_domain_points

    if N > m:
        Sigma = (V.T @ V) / (N - 2)  # mxm
    else:
        Sigma = (V @ V.T) / (N - 2)  # NxN

    if method == "L2-Simul":
        q = 1

        v_1j = y1.T - np.mean(y1, axis=0, keepdims=True).T
        v_2j = y2.T - np.mean(y2, axis=0, keepdims=True).T

        n_array = [n1, n2]
        v_array = [v_1j, v_2j]
        LHS = 0

        ts = set_up_time_bar('Calculating Simulated Null Distribution', sum(n_array))

        for ii in range(2):
            n_i = n_array[ii]
            V_group = v_array[ii]
            for jj in range(n_i):
                v_ij = V_group[:, jj]
                LHS += np.outer(v_ij, v_ij) @ np.outer(v_ij, v_ij)
                ts.progress()

        ts.delete()

        LHS = LHS / N

        if LHS.shape == Sigma.shape:
            omega_hat = LHS - (Sigma @ Sigma)  # matrix multiplication
        else:
            SigmaLarge = ((n1 - 1) * np.cov(y1, rowvar=False) + (n2 - 1) * np.cov(y2, rowvar=False)) / (N - 2)
            omega_hat = LHS - (SigmaLarge @ SigmaLarge)  # matrix multiplication

        eig_gamma_hat = np.real(np.linalg.eigvals(omega_hat))
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        T_null = chi_sq_mixture(q, eig_gamma_hat, self.n_simul)
        kde = gaussian_kde(T_null)
        pvalue = 1 - kde.integrate_box_1d(-np.inf, stat)
        pvalue = max(0,min(1,pvalue))

    elif method == "L2-BiasReduced":  # Bias Reduced
        A = np.trace(Sigma @ Sigma) + np.trace(Sigma)**2
        B = 2 * np.trace(np.linalg.matrix_power(Sigma, 4)) + 2 * np.trace(Sigma @ Sigma)**2

        alpha = (N - 2)**2 / (N * (N - 3)) * (B - A**2 / (N - 2)) / A
        df = (1 + 1 / (N - 2)) * (A**2 - 2 * B / (N - 1)) / (B - A**2 / (N - 2))

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "L2-Naive":  # Naive Method
        an = np.trace(Sigma)
        bn = np.trace(Sigma @ Sigma)
        cn = np.trace(np.linalg.matrix_power(Sigma, 3))
        dn = np.trace(np.linalg.matrix_power(Sigma, 4))

        An = an
        Bn = (N - 2)**2 / (N * (N - 3)) * (bn - An**2 / (N - 2))
        Cn = (cn - 3 / (N - 2) * Bn * An) / (1 + 3 / (N - 2))
        Dn = (dn - 6 / (N - 2) * Cn * An) / (1 + 6 / (N - 2))
        A = Bn + An**2
        B = 2 * Dn + 2 * Bn**2
        alpha = B / A
        df = A**2 / B

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "Bootstrap-Test":  # Bootstrap test
        ts = set_up_time_bar('Running Bootstrap Test', self.n_boot)
        vstat = np.zeros(self.n_boot)
        k = 2

        def bootstrap_iteration(ii):
            flag1 = np.random.choice(n1, n1, replace=True)
            flag2 = np.random.choice(n2, n2, replace=True)

            yy1 = y1[flag1, :]
            yy2 = y2[flag2, :]

            # Computing p_s : Bootstrap Pooled Covariance
            mu1 = np.mean(yy1, axis=0)
            R1 = yy1 - np.ones((n1, 1)) @ mu1.reshape(1, -1)

            mu2 = np.mean(yy2, axis=0)
            R2 = yy2 - np.ones((n2, 1)) @ mu2.reshape(1, -1)

            R = np.vstack([R1, R2])

            if N > m:
                p_s = (R.T @ R) / (N - k)  # p x p pooled covariance matrix
            else:
                p_s = (R @ R.T) / (N - k)

            stat0 = 0
            nni = 0
            for i in range(k):
                ni = gsize[i]
                flag = np.arange(nni, nni + ni)
                Ri = R[flag, :]

                # ith group's covariance
                if N > m:
                    p_si = (Ri.T @ Ri) / (ni - 1)  # Vi: ni x p
                    temp = np.trace(np.linalg.matrix_power(p_si - p_s, 2))
                else:
                    p_si = (Ri @ Ri.T) / (ni - 1)
                    temp = (np.trace(np.linalg.matrix_power(p_si, 2)) -
                           2 * np.trace(((Ri @ R.T) @ R) @ Ri.T) / (N - k) / (ni - 1) +
                           np.trace(np.linalg.matrix_power(p_s, 2)))

                stat0 += (ni - 1) * temp
                nni += ni

            return stat0

        # Bootstrap iterations
        for ii in range(self.n_boot):
            vstat[ii] = bootstrap_iteration(ii)
            ts.progress()

        ts.delete()
        pvalue = np.mean(vstat > stat)

    elif method == "Permutation-Test":  # Permutation test
        ts = set_up_time_bar('Running Permutation Test', self.N_permutations)
        vstat = np.zeros(self.N_permutations)
        k = 2
        Y = np.vstack([y1, y2])

        def permutation_iteration(ii):
            flag = np.random.permutation(N)
            Y_perm = Y[flag, :]
            yy1 = Y_perm[:n1, :]
            yy2 = Y_perm[n1:, :]

            mu1 = np.mean(yy1, axis=0)
            R1 = yy1 - np.ones((n1, 1)) @ mu1.reshape(1, -1)

            mu2 = np.mean(yy2, axis=0)
            R2 = yy2 - np.ones((n2, 1)) @ mu2.reshape(1, -1)

            R = np.vstack([R1, R2])

            if N > m:
                p_s = (R.T @ R) / (N - k)
            else:
                p_s = (R @ R.T) / (N - k)

            stat0 = 0
            nni = 0
            for i in range(k):
                ni = gsize[i]
                flag = np.arange(nni, nni + ni)
                Ri = R[flag, :]

                if N > m:
                    p_si = (Ri.T @ Ri) / (ni - 1)
                    temp = np.trace(np.linalg.matrix_power(p_si - p_s, 2))
                else:
                    p_si = (Ri @ Ri.T) / (ni - 1)
                    temp = (np.trace(np.linalg.matrix_power(p_si, 2)) -
                           2 * np.trace(((Ri @ R.T) @ R) @ Ri.T) / (N - k) / (ni - 1) +
                           np.trace(np.linalg.matrix_power(p_s, 2)))

                stat0 += (ni - 1) * temp
                nni += ni

            return stat0

        # Permutation iterations
        for ii in range(self.N_permutations):
            vstat[ii] = permutation_iteration(ii)
            ts.progress()

        ts.delete()
        pvalue = np.mean(vstat > stat)

    return pvalue

# TODO: Needs work
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

# TODO: Needs work
def two_group_cov(self, method, y1, y2):
    n1, L = y1.shape
    n2, _ = y2.shape

    Sigma1 = np.cov(y1, rowvar=False)
    Sigma2 = np.cov(y2, rowvar=False)
    N = n1 + n2

    Sigma = ((n1 - 1) * Sigma1 + (n2 - 1) * Sigma2) / (N - 2)
    stat = (n1 - 1) * (n2 - 1) / (N - 2) * np.trace((Sigma1 - Sigma2) @ (Sigma1 - Sigma2))

    if method == "L2-Simul":
        q = 1
        v_1j = y1.T - np.mean(y1, axis=0, keepdims=True).T
        v_2j = y2.T - np.mean(y2, axis=0, keepdims=True).T

        n_array = [n1, n2]
        v_array = [v_1j, v_2j]
        LHS = 0

        for ii in range(2):
            n_i = n_array[ii]
            V = v_array[ii]
            for jj in range(n_i):
                v_ij = V[:, jj]
                LHS += np.outer(v_ij, v_ij) @ np.outer(v_ij, v_ij)

        LHS /= N

        if LHS.shape == Sigma.shape:
            omega_hat = LHS - (Sigma @ Sigma)
        else:
            SigmaLarge = ((n1 - 1) * np.cov(y1, rowvar=False) + (n2 - 1) * np.cov(y2, rowvar=False)) / (N - 2)
            omega_hat = LHS - (SigmaLarge @ SigmaLarge)

        eig_gamma_hat = np.real(np.linalg.eigvals(omega_hat))
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        T_null = utils.chi_sq_mixture(q, eig_gamma_hat, self.n_simul)

        kde = gaussian_kde(T_null)
        pvalue = 1 - kde.integrate_box_1d(-np.inf, stat)
        pvalue = max(0,min(1,pvalue))

    elif method == "L2-BiasReduced":
        A = np.trace(Sigma @ Sigma) + np.trace(Sigma)**2
        B = 2 * np.trace(np.linalg.matrix_power(Sigma, 4)) + 2 * np.trace(Sigma @ Sigma)**2

        alpha = (N - 2)**2 / (N * (N - 3)) * (B - A**2 / (N - 2)) / A
        df = (1 + 1 / (N - 2)) * (A**2 - 2 * B / (N - 1)) / (B - A**2 / (N - 2))

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "L2-Naive":
        an = np.trace(Sigma)

        bn = np.trace(Sigma @ Sigma)
        cn = np.trace(np.linalg.matrix_power(Sigma, 3))
        dn = np.trace(np.linalg.matrix_power(Sigma, 4))

        An = an
        Bn = (N - 2)**2 / (N * (N - 3)) * (bn - An**2 / (N - 2))
        Cn = (cn - 3 / (N - 2) * Bn * An) / (1 + 3 / (N - 2))
        Dn = (dn - 6 / (N - 2) * Cn * An) / (1 + 6 / (N - 2))

        A = Bn + An**2
        B = 2 * Dn + 2 * Bn**2
        alpha = B / A
        df = A**2 / B
        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "Bootstrap-Test":
        vstat = np.zeros(self.n_boot)

        for ii in range(self.n_boot):
            flag1 = np.random.choice(n1, n1, replace=True)
            flag2 = np.random.choice(n2, n2, replace=True)

            yy1 = y1[flag1, :]
            yy2 = y2[flag2, :]

            S1 = np.cov(yy1, rowvar=False)
            S2 = np.cov(yy2, rowvar=False)

            stat0 = (n1 - 1) * (n2 - 1) / (N - 2) * np.trace(np.linalg.matrix_power((S1 - S2) - (Sigma1 - Sigma2), 2))
            vstat[ii] = stat0
        pvalue = np.mean(vstat > stat)

    elif method == "Permutation-Test":
        vstat = np.zeros(self.N_permutations)
        y1_centered = y1 - np.mean(y1, axis=0)
        y2_centered = y2 - np.mean(y2, axis=0)
        yy = np.vstack((y1_centered, y2_centered))

        for ii in range(self.N_permutations):
            flag = np.random.permutation(N)

            yy1 = yy[flag[:n1], :]
            yy2 = yy[flag[n1:], :]

            S1 = np.cov(yy1, rowvar=False)
            S2 = np.cov(yy2, rowvar=False)

            stat0 = (n1 - 1) * (n2 - 1) / (N - 2) * np.trace(np.linalg.matrix_power(S1 - S2, 2))
            vstat[ii] = stat0

        pvalue = np.mean(vstat > stat)

    return pvalue
