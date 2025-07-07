import numpy as np
from scipy.stats import chi2, gaussian_kde
from scipy.linalg import eig

def two_group_cov(self, method, y1, y2):
    n1, L = y1.shape
    n2, _ = y2.shape

    Sigma1 = np.cov(y1, rowvar=False)
    Sigma2 = np.cov(y2, rowvar=False)
    N = n1 + n2

    Sigma = ((n1 - 1) * Sigma1 + (n2 - 1) * Sigma2) / (N - 2)
    stat = (n1 - 1) * (n2 - 1) / (N - 2) * np.trace(np.dot((Sigma1 - Sigma2), (Sigma1 - Sigma2)))

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
                LHS += np.outer(v_ij, v_ij) * np.outer(v_ij, v_ij)

        LHS /= N

        if LHS.shape == Sigma.shape:
            omega_hat = LHS - np.dot(Sigma, Sigma)
        else:
            SigmaLarge = ((n1 - 1) * np.cov(y1, rowvar=False) + (n2 - 1) * np.cov(y2, rowvar=False)) / (N - 2)
            omega_hat = LHS - np.dot(SigmaLarge, SigmaLarge)

        eig_gamma_hat = np.real(eig(omega_hat, right=False))
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

        T_null = self.__class__.chi_sq_mixture(q, eig_gamma_hat, self.N_simul)

        # kde = KernelDensity(kernel='gaussian').fit(T_null.reshape(-1, 1))
        # pvalue = 1 - np.exp(kde.score_samples(np.array([stat]).reshape(-1, 1)))[0]

        kde = gaussian_kde(T_null)
        pvalue = 1 - kde.integrate_box_1d(-np.inf, stat)

    elif method == "L2-BiasReduced":
        A = np.trace(np.linalg.matrix_power(Sigma, 2)) + np.trace(Sigma)**2
        B = 2 * np.trace(np.linalg.matrix_power(Sigma, 4)) + 2 * np.trace(np.linalg.matrix_power(Sigma, 2))**2

        alpha = (N - 2)**2 / (N * (N - 3)) * (B - A**2 / (N - 2)) / A
        df = (1 + 1 / (N - 2)) * (A**2 - 2 * B / (N - 1)) / (B - A**2 / (N - 2))

        pvalue = 1 - chi2.cdf(stat / alpha, df)

    elif method == "L2-Naive":
        an = np.trace(Sigma)

        bn = np.trace(np.linalg.matrix_power(Sigma, 2))
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
        vstat = np.zeros(self.N_boot)

        for ii in range(self.N_boot):
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
