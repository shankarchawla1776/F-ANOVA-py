from functionalANOVA.core import utils
import numpy as np
from scipy import stats
from scipy.stats import chi2, f, ncx2
from scipy.linalg import inv, sqrtm
from tqdm import tqdm


# TODO: Needs work
def run_oneway(self, eig_gamma_hat, eta_i, params, H0):

    n_methods = len(self._methods.anova_methods_used)

    pvalue_matrix = np.zeros((H0.n_tests, n_methods))

    # Utilized for Future Work
    # self.CriticalValues = [[None, None, None] for _ in range(n_methods)]

    counter = 0

    for method in self._methods.anova_methods_used:
        counter += 1

        match method:
         case "L2-Simul":
            T_null = utils.chi_sq_mixture(H0.q, eig_gamma_hat, self.n_simul)

            T_NullFitted = stats.gaussian_kde(T_null)

            p_value = np.zeros((H0.n_tests, 1))

            for j in range(H0.n_tests):
                p_value[j] = 1 - T_NullFitted.integrate_box_1d(-np.inf, params.T_n[j])
                p_value[j] = max(0,min(1,p_value[j]))

                if self.show_simul_plots:
                    plot_test_stats(p_value[j], self.alpha, T_null, params.T_n[j], method + " test", self.hypothesis, H0.pair_vec[j])

            pvalue_matrix[:, counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][1] = np.quantile(T_null, 1-self.alpha)

            if self.hypothesis == "FAMILY":
                utils.update_family_table(self._tables.oneway, method, [T_NullFitted])

         case "L2-Naive":
            p_value = np.zeros((H0.n_tests, 1))

            for j in range(H0.n_tests):
                p_value[j] = 1 - ncx2.cdf(params.T_n[j] / params.beta_hat, H0.q * params.kappa_hat, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][1] = params.beta_hat * ncx2.ppf(1 - self.alpha, H0.q * params.kappa_hat, 0)

         case "L2-BiasReduced":
            p_value = np.zeros((H0.n_tests,1))
            for j in range(H0.n_tests):
                p_value[j] = 1 - ncx2.cdf(params.T_n[j] / params.beta_hat_unbias, H0.q * params.kappa_hat_unbias, 0)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][1] = params.beta_hat_unbias * ncx2.ppf(1 - self.alpha, H0.q * params.kappa_hat_unbias,0)

         case "L2-Bootstrap":
            T_n_Boot = np.zeros((self.n_boot, H0.n_tests))

            match self.hypothesis:
                case "FAMILY":
                    self._labels.hypothesis = H0.pair_vec[0]

                    yy = np.array([])

                    for j in range(self._groups.k):
                        # yy = np.append(yy, self.data[j].T)
                        if j == 0:
                            yy = self.data[j].T
                        else:
                            yy = np.vstack([yy, self.data[j].T])

                    T_n_Boot= utils.l2_bootstrap(self, yy,  method)

                    utils.update_family_table(self._tables.oneway, method, [self.n_boot])
                
                case "PAIRWISE":
                    # n_tests is the number of tests; n = self.N is real n
                    for j in range(H0.n_tests):
                        self._labels.hypothesis = H0.pair_vec[j]
                        Ct = H0.C[j, :]
                        for k in tqdm(range(self.n_boot), desc=self._setup_time_bar(method)):
                            eta_i_star, _, _ = utils.group_booter(self.data, self.n_domain_points, self._groups.k, self.n_i, self.N)
                            rh_side = Ct @ H0.D @ Ct.T
                            if rh_side.ndim == 0:
                                SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * 1/rh_side
                            else:
                                SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * inv(rh_side)
                            T_n_Boot[k,j] = np.sum(SSH_t)

            p_value = np.zeros((H0.n_tests, 1))
            crit_vals = np.zeros((H0.n_tests, 1))
            for j in range(H0.n_tests):
                p_value[j] = np.mean(T_n_Boot[:,j] >= params.T_n[j])
                crit_vals[j] = np.quantile(T_n_Boot[:,j], 1 - self.alpha)

            pvalue_matrix[:,counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][1] = crit_vals[-1]

         case "F-Simul":

            ratio = (self.N - self._groups.k) / H0.q
            T_null = utils.chi_sq_mixture(H0.q, eig_gamma_hat, self.n_simul)
            F_null_denom = utils.chi_sq_mixture(self.N - self._groups.k, eig_gamma_hat, self.n_simul)
            F_null = (T_null / F_null_denom) * ratio
            F_NullFitted = stats.gaussian_kde(F_null)

            p_value = np.zeros((H0.n_tests, 1))

            for j in range(H0.n_tests):
                p_value[j] = 1 - F_NullFitted.integrate_box_1d(-np.inf, params.F_n[j])
                p_value[j] = max(0,min(1,p_value[j]))
                
                if self.show_simul_plots:
                    plot_test_stats(p_value[j], self.alpha, F_null, params.F_n[j], method + " test", self.hypothesis, H0.pair_vec[j])

            pvalue_matrix[:, counter-1] = p_value.flatten()

            if self.hypothesis == "FAMILY":
                utils.update_family_table(self._tables.oneway, method, [F_NullFitted])

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][2] = np.quantile(F_null, 1 - self.alpha)
            # self.CriticalValues[counter-1][1] = np.quantile((self.CriticalValues[counter-1][2] / ratio) * F_null_denom, 1 - self.alpha)

         case "F-Naive":
            p_value = np.zeros((H0.n_tests,1))
            for j in range(H0.n_tests):
                p_value[j] = 1 - f.cdf(params.F_n[j], H0.q * params.kappa_hat, (self.N - self._groups.k) * params.kappa_hat)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            A = H0.q * params.kappa_hat
            B = (self.N - self._groups.k) * params.kappa_hat

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][2] = f.ppf(1 - self.alpha, A, B)

         case "F-BiasReduced":
            p_value = np.zeros((H0.n_tests, 1))

            for j in range(H0.n_tests):
                p_value[j] = 1 - f.cdf(params.F_n[j],
                                       H0.q * params.kappa_hat_unbias,
                                       (self.N - self._groups.k) * params.kappa_hat_unbias)

            pvalue_matrix[:, counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][2] = f.ppf(1 - self.alpha, H0.q * params.
            #                                           kappa_hat_unbias,
            #                                           (self.N - self._groups.k) * params.kappa_hat_unbias)

         case "F-Bootstrap":
            F_n_Boot = np.zeros((self.n_boot, H0.n_tests))
            ratio = (self.N - self._groups.k) / H0.q
            crit_vals = np.zeros((H0.n_tests, 1))
            ReversedT_n = np.zeros((H0.n_tests, 1))

            match self.hypothesis:
                case "FAMILY":
                    self._labels.hypothesis = H0.pair_vec[0]
                    f_n_Denominator_Boot = np.nan * np.ones((self.n_boot, H0.n_tests))
                    yy = np.array([])

                    for j in range(self._groups.k):
                        # yy = np.append(yy, self.data[j].T)

                        if j == 0:
                            yy = self.data[j].T
                        else:
                            yy = np.vstack([yy, self.data[j].T])
                    F_n_Boot = utils.f_bootstrap(self, yy, method)
                    utils.update_family_table(self._tables.oneway, method, [self.n_boot])

                case "PAIRWISE":
                    f_n_Denominator_Boot = np.zeros((self.n_boot, H0.n_tests))

                    for j in range(H0.n_tests):
                        self._labels.hypothesis  = H0.pair_vec[j]
                        Ct = H0.C[j, :]

                        for k in tqdm(range(self.n_boot), desc=self._setup_time_bar(method)):
                            eta_i_star, _, gamma_hat_star = utils.group_booter(self.data, self.n_domain_points, self._groups.k, self.n_i, self.N)
                            f_n_Denominator_Boot[k,j] = np.trace(gamma_hat_star) * (self.N - self._groups.k)
                            
                            rh_side = Ct @ H0.D @ Ct.T
                            
                            if rh_side.ndim == 0:
                                SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * 1/rh_side
                            else:
                                SSH_t = ((Ct @ (eta_i_star - eta_i).T)**2) * inv(rh_side)

                            T_n_Boot = np.sum(SSH_t)
                            F_n_Boot[k,:] = (T_n_Boot / f_n_Denominator_Boot[k,j]) * ratio
                            
                case _:
                    raise ValueError(f'Unsupported hypothesis: {self.hypothesis}')

            p_value = np.zeros((H0.n_tests, 1))

            for j in range(H0.n_tests):
                p_value[j] = 1 - np.sum(F_n_Boot[:,j] < params.F_n[j]) / float(self.n_boot)

                crit_vals[j] = np.quantile(F_n_Boot[:,j], 1 - self.alpha)
                ReversedT_n[j] = np.quantile((crit_vals[j] / ratio) * f_n_Denominator_Boot[:,j], 1 - self.alpha)

            pvalue_matrix[:,counter-1] = p_value.flatten()

            # self.CriticalValues[counter-1][0] = method
            # self.CriticalValues[counter-1][1] = ReversedT_n[j]
            # self.CriticalValues[counter-1][2] = crit_vals[j]


    return pvalue_matrix


# Completed
def run_onewayBF(self, method, data, contrast, c, indicator_a=None):

    N = self.N
    p = self.n_domain_points

    if indicator_a is None:
        gsize = self.n_i
        aflag = utils.aflag_maker(gsize)

    else:
        aflag = indicator_a


    yy = data
    contrast = np.array(contrast)  # Only if it's not already an array

    aflag0 = np.unique(aflag)

    k = len(aflag0)
    mask = np.ones(k, dtype=bool)

    vmu = np.array([])
    A = np.array([])
    A2 = np.array([])
    B2 = np.array([])
    K2b = None
    gsize = np.zeros(k)
    S_ii = [None] * k
    pstat = [None, None]
    pvalue = np.nan
    stat = np.nan

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
            else:
                raise ValueError(f"Unsupported method: {method}")
            A2 = np.append(A2, A2i)
            B2 = np.append(B2, B2i)

    D = np.diag(1/gsize)
    
    if contrast.ndim == 2:
        H = np.array(sqrtm((inv(contrast @ D @ contrast.T))))
        stat0 = np.trace(H @ (contrast @ vmu - c) @ (contrast @ vmu-c).T @ H.T)
    else:
        H = np.sqrt(np.divide(1.0, contrast @ D @ contrast.T)) 
        stat0 = np.multiply(H, (contrast @ vmu - c)) @ np.multiply( (contrast @ vmu-c).T, H.T)
    

    if method in ["L2-Naive", "L2-BiasReduced", "F-Naive", "F-BiasReduced"]:
        Dh = np.sqrt(D)
        
        if contrast.ndim == 2:
            W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh
        else:
            W = np.multiply( (Dh @ contrast.reshape(-1, 1)), H) @ np.multiply(H, contrast.T @ Dh).reshape(1, -1)

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

    elif method in ["F-Bootstrap", "F-Simul"]:
        Dh = np.sqrt(D)
        # k by k
        
        if contrast.ndim == 2:
            W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh
        else:
            W = np.multiply( (Dh @ contrast.reshape(-1, 1)), H) @ np.multiply(H, contrast.T @ Dh).reshape(1, -1)

        dd = np.diag(W)
        K1 = np.sum(dd * A)

        f_stat = stat0 / K1

        if self.hypothesis == "FAMILY":

            b_n = np.sqrt(gsize)
            A_n = np.eye(k) - np.outer(b_n, b_n) / N
            A_n_ii = np.diag(A_n)

            mask = np.ones(k, dtype=bool)

        elif self.hypothesis == "PAIRWISE":
            # # np.logical_not makes a boolean mask where true means 0 in contrast. replace with np.any
            # mask = np.logical_not(np.abs(contrast.T))

            mask = (contrast != 0)
            g_n = gsize[mask]

            N_n = np.sum(g_n)
            k_n = len(g_n)

            b_n = np.sqrt(g_n)
            A_n = np.eye(k_n) - np.outer(b_n, b_n) / N_n
            A_n_ii = np.diag(A_n)

        elif self.hypothesis in ['INTERACTION', 'PRIMARY', 'SECONDARY']:

            A_n = D**0.5 @ contrast.T @ inv(contrast @ D @ contrast.T) @ contrast @ D**0.5
            A_n_ii = np.diag(A_n)
            mask = np.ones(k, dtype=bool)
            
        else:
            raise ValueError(f"Unsupported hypothesis: {self.hypothesis}")
        
        if method == "F-Bootstrap":
            Bstat = np.zeros(self.n_boot)

            for ii in tqdm(range(self.n_boot), desc=self._setup_time_bar(method)):
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

                if contrast.ndim == 2:
                    temp = H @ contrast @ (Bmu - vmu)
                    T_n = np.trace(temp @ temp.T)
                else:
                    temp = np.multiply(H, contrast) @ (Bmu - vmu)
                    T_n = temp.reshape(1, -1) @ temp.reshape(-1, 1)

                S_n = np.sum(A_n_ii * tr_gamma)
                temp = T_n / S_n

                Bstat[ii] = temp


            pvalue = np.mean(Bstat > f_stat)
            pstat = [f_stat, pvalue]
            
        elif method == "F-Simul":
            Dh = np.sqrt(D) # kxk
            
            if contrast.ndim == 2:
                W = Dh @ contrast.T @ H.T @ H @ contrast @ Dh
            else:
                W = np.multiply( (Dh @ contrast.reshape(-1, 1)), H) @ np.multiply(H, contrast.T @ Dh).reshape(1, -1)

            dd = np.diag(W)
            K1 = np.sum(dd * A)
            f_stat = stat0 / K1

            if self.hypothesis in ['FAMILY', 'PAIRWISE']:
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

                eig_gamma_hat = np.linalg.eigvalsh(COV_Sum)
                eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

                q = k_n - 1
                T_null = utils.chi_sq_mixture(q, eig_gamma_hat, self.n_simul)

                S_null = np.zeros(self.n_simul)
                S_ii_subset = np.asarray([S_ii[i] for i in range(k) if mask[i]])

                for i in range(k_n):
                    eig_gamma_hat = np.linalg.eigvalsh(S_ii_subset[i])
                    eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

                    S_temp = utils.chi_sq_mixture(int(g_n[i]) - 1, eig_gamma_hat, self.n_simul)
                    S_temp = (S_temp * A_n_ii[i]) / (g_n[i] - 1)
                    S_null += S_temp

                F_null = T_null / S_null
                F_NullFitted = stats.gaussian_kde(F_null)

                pvalue = 1 - F_NullFitted.integrate_box_1d(-np.inf, f_stat)
                pvalue = max(0,min(1,pvalue))

                pstat = [f_stat, pvalue]
            else:
                pstat = [f_stat, np.nan]

        stat = pstat[0]
        pvalue = pstat[1]

    elif method == "L2-Bootstrap":
        Bstat = np.zeros(self.n_boot)

        for ii in tqdm(range(self.n_boot), desc=self._setup_time_bar(method)):

            Bmu = np.empty((0, p))
            for i in range(k):
                iflag = (aflag == aflag0[i])
                yi = yy[iflag, :]
                ni = int(gsize[i])

                Bflag = np.random.choice(ni, ni, replace=True)
                Byi = yi[Bflag, :]
                Bmui = np.mean(Byi, axis=0)
                Bmu = np.vstack([Bmu, Bmui])

            if contrast.ndim == 2:
                temp = H @ contrast @ (Bmu - vmu)
                temp = np.trace(temp @ temp.T)
            else:
                temp = np.multiply(H, contrast) @ (Bmu - vmu)
                temp = temp.reshape(1, -1) @ temp.reshape(-1, 1)
                
            Bstat[ii] = temp


        pvalue = np.mean(Bstat > stat0)
        pstat = [stat0, pvalue]

    elif method == "L2-Simul":
        if self.hypothesis in ['FAMILY', 'PAIRWISE']:
            build_covar_star = np.zeros((self.n_domain_points, 0))

            if contrast.ndim == 2:
                mask = np.any(contrast.T.astype(bool), axis=1)
            else:
                mask = contrast.T.astype(bool)
            
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

            eig_gamma_hat = np.linalg.eigvalsh(COV_Sum)
            eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]

            q = k_n - 1
            T_null = utils.chi_sq_mixture(q, eig_gamma_hat, self.n_simul)

            T_NullFitted = stats.gaussian_kde(T_null)
            pvalue = 1 - T_NullFitted.integrate_box_1d(-np.inf, stat0)
            pvalue = max(0,min(1,pvalue))
            pstat = [stat0, pvalue]


    else:
        raise ValueError(f'Unknown Method: {method}')
    
    return pstat[1], pstat[0]

