import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Any, ClassVar, cast, Sequence
from dataclasses import dataclass, field
import warnings
from functionalANOVA.core import utils
from functionalANOVA.core.plot_means import plot_means
from scipy import stats
from scipy.stats import chi2, f
from scipy.linalg import inv, sqrtm
from tqdm import tqdm

@dataclass  # class to store these labels
class ANOVALabels:
    group: Optional[List[str]] = None
    primary: Optional[List[str]] = None
    secondary: Optional[List[str]] = None
    generic_group: Optional[bool] = True
    
    # unit Labels 
    domain:Optional[str] = None  # such as Time
    response:Optional[str] = None
    
    # Hypothesis Labels
    hypothesis: Optional[List[str]] = None
    H0_OneWay:ClassVar[Tuple[str, ...]] = ("FAMILY", "PAIRWISE")
    H0_TwoWay:ClassVar[Tuple[str, ...]] = ("FAMILY", "PAIRWISE", "INTERACTION", "PRIMARY", "SECONDARY", "CUSTOM")
    
@dataclass  # class to store units
class ANOVAUnits:
    domain: Optional[str] = None  # Such as seconds
    response: Optional[str] = None
    
@dataclass
class ANOVATables:
    oneway: Optional[pd.DataFrame] = None
    oneway_bf: Optional[pd.DataFrame] = None
    twoway: Optional[pd.DataFrame] = None
    twoway_bf: Optional[pd.DataFrame] = None
    covar: Optional[pd.DataFrame] = None
    sw: Optional[pd.DataFrame] = None
    sig_figs = 4  # Significant Figures to display 
    
@dataclass
class ANOVAMethods:  # clas to store methods
    # Class-level constants (shared by all instances, not settable per instance)
    anova_methods: ClassVar[Tuple[str, ...]] = (
        "L2-Simul", "L2-Naive", "L2-BiasReduced", "L2-Bootstrap",
        "F-Simul", "F-Naive", "F-BiasReduced", "F-Bootstrap"
    )

    covar_methods: ClassVar[Tuple[str, ...]] = (
        "L2-Simul", "L2-Naive", "L2-BiasReduced", "Permutation-Test", "Bootstrap-Test"
    )
    # Instance fields (can be changed internally if needed)
    anova_methods_used: Tuple[str, ...] = ()
    covar_methods_used: Tuple[str, ...] = ()
    
@dataclass
class ANOVAGroups:
    
    k: int = 0       # One-way ANOVA: number of groups
    A: int = 0       # Two-way ANOVA: primary levels
    B: int = 0       # Two-way ANOVA: secondary levels
    AB: int = 0      # Two-way ANOVA: total combinations
    
    subgroup_indicator: Union[np.ndarray, List[np.ndarray], None] = None # indicator Array for B
    
    contrast: Optional[np.ndarray] = None # User Specified Constrast vector
    contrast_factor:  Optional[int] = None # Either 1 for Primary, 2 for Secondary

class functionalANOVA():
    
    @property
    def tables(self) -> ANOVATables:
        return self._tables

    @property
    def methods(self) -> ANOVAMethods:
        return self._methods
    
    @property
    def labels(self) -> ANOVALabels:
        return  self._labels
    
    @property
    def units(self) -> ANOVAUnits:
        return  self._units
    
    @property
    def groups(self) -> ANOVAGroups:
        return  self.groups
    
    def __init__(
        self,
        data_list: List[np.ndarray] | Tuple[np.ndarray],
        d_grid: np.ndarray,
        grid_bounds: Tuple[float, float],
        n_boot: int = 10_000,
        n_simul: int = 10_000,
        alpha: float = 0.05,
        subgroup_indicator: Union[np.ndarray, List[np.ndarray], None] = None,
        group_labels: Optional[List[str]] = None,
        primary_labels: Optional[List[str]] = None,
        secondary_labels: Optional[List[str]] = None,
        domain_units_label: Optional[str] = None,
        response_units_label: Optional[str] = None
    ):
        
     
        self.data = []
        self.grid_bounds = grid_bounds
        self.d_grid = d_grid
        self.n_boot = int(n_boot)
        self.n_simul = int(n_simul)
        self.alpha = alpha
        self._labels = ANOVALabels(group_labels, primary_labels, secondary_labels)
        self._units = ANOVAUnits(domain_units_label, response_units_label)
        self._tables = ANOVATables() 
        self._methods = ANOVAMethods()
        self._groups = ANOVAGroups(subgroup_indicator=subgroup_indicator)
        
      
        # Public and writable fields
        self.weights = "proportional"
        self.hypothesis = "FAMILY"
        self.verbose = True
        self.show_simul_plots = False # Shows Null distribution plots for "Simul" Methods
        
        # Validate All Inputs 
        self._validate_instantiation_inputs()
        
        self._groups.k = len(data_list)
        self.n_i = tuple(x.shape[1] for x in data_list)
        self.n_ii = None
        self.N = sum(self.n_i)    # Total Samples combined
        
        n_rows_per_group = [x.shape[0] for x in data_list]
        expected_length = self.d_grid.shape[0]

        if not all(n == expected_length for n in n_rows_per_group):
            raise ValueError("All groups/replicates must have the same vector length as the domain")
        
        self._function_subsetter()
        
        self.n_domain_points = len(range(self.lb_index, self.ub_index + 1))

        if self.n_domain_points < 1000:
            warnings.warn(f'Functional data has a resolution of {self.n_domain_points} elements. It is recommended to have a resolution of at least 1000 elements for the convergence of the F-ANOVA p-values')
            
        # Subset  data
        for k in range(self._groups.k):
            self.data.append(data_list[k][self.lb_index : self.ub_index + 1, :]) # exclusive at the upper bound
        
        if not self._groups.subgroup_indicator: # One Way ANOVA set up
            if  self._labels.group:
                assert len(self._labels.group) == self._groups.k, "Each Group Must Have Exactly One Label Associated to it"
                self._labels.generic_group = False
            else:
                self._labels.group = [f"{i+1}" for i in range(self._groups.k)] #Automatically Assign Group Labels
                self._labels.generic_group = True
        else:
            self._set_up_two_way()  # Creates Indicator Matrices and default Labels
            self._n_ii_generator()  # Creates Secondary Size Array

    # def plot_means(self):
    #     #TODO Migrate and integrate plotting method here
    #     pass

    # def plot_covariances(self):
    #     #TODO Migrate and integrate plotting method here
    #     pass
    
    def oneway_bf(self,
                  n_boot: int = 10_000,
                  n_simul: int = 10_000,
                  alpha: float = 0.05,
                  methods: Optional[Sequence[str]] = None,
                  hypothesis: Optional[Sequence[str]] = None):
        
        # Sometype of checking for inputs above
        self._validate_stat_inputs(alpha, n_boot, n_simul, methods, hypothesis)
            
        n_methods = len(self._methods.anova_methods_used)
        
        yy = np.vstack([arr.T for arr in self.data])

        pair_vec = []
        
        # Set up Hypothesis
        match self.hypothesis:
            case "PAIRWISE":
                C = utils.construct_pairwise_contrast_matrix(self._groups.k)
                n_tests = C.shape[0]
                if self._labels.group is None:
                    raise ValueError("Group labels must be provided.")
                if len(self._labels.group) != self._groups.k:
                    raise ValueError(
                        f"Each group must have exactly one label. "
                        f"Got {len(self._labels.group)} labels for {self._groups.k} groups.")
                for cc in range(n_tests):
                    # Find the indices where C[cc, :] is True
                    idx = np.where(C[cc])[0]
                    t1, t2 = self._labels.group[idx[0]], self._labels.group[idx[1]]
                    pair_label = f"{t1} & {t2}"
                    pair_vec.append(pair_label)
                    
            case "FAMILY":
                n_tests = 1
                pair_vec.append("FAMILY")
                k = self._groups.k
                C = np.hstack([np.eye(k - 1), -np.ones((k - 1, 1))])
                
                self._tables.oneway_bf = pd.DataFrame({
                                    'Family-Wise Method': self._methods.anova_methods_used,
                                    'Test-Statistic': [np.nan] * n_methods,
                                    'P-Value': [np.nan] * n_methods,
                                    'Verdict': [None] * n_methods  # Empty strings for verdict (like MATLAB strings)
                                })
            case _:
                raise ValueError("Unknown Hypothesis provided")
        
        # Iterate over Methods
        # Convert pair_vec to a DataFrame
        T_hypothesis = pd.DataFrame({'Hypothesis': pair_vec})

        # Create NaN matrices
        p_value_matrix = np.full((n_tests, n_methods), np.nan)
        test_stat = np.full((1, n_methods), np.nan)
        
        counter = 0
        c = 0  # Assuming Equality to each other and not a generic constant
        statistic = np.nan
        
        for method in  self._methods.anova_methods_used:
            p_value = np.zeros(n_tests)

            for ii in range(n_tests):
                if self.hypothesis == 'FAMILY':
                    C_input = C
                elif self.hypothesis == 'PAIRWISE':
                    C_input = C[ii, :]
                    self._labels.hypothesis = pair_vec[ii]
                else:
                    raise ValueError(f"Unsupported hypothesis type: {self.hypothesis}")

                p_value[ii], statistic = self._run_onewayBF(method, yy, C_input, c)

            p_value_matrix[:, counter] = p_value
            test_stat[0, counter] = statistic
            counter += 1

        self._prep_tables('oneway_bf', p_value_matrix, T_hypothesis, test_stat)


        if self.verbose:
            self._data_summary_report_one_way(ANOVA_TYPE='heteroskedastic')
            self._show_table(self._tables.oneway_bf)

    def _run_onewayBF(self, method, data, contrast, c, indicator_a=None):

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

                for ii in tqdm(range(self.n_boot), desc=self._set_up_time_bar(method)):
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

            for ii in tqdm(range(self.n_boot), desc=self._set_up_time_bar(method)):

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

    def _prep_tables(self, anova_method, p_value_matrix, T_hypothesis, test_stat):
        match anova_method:
            case 'oneway':
                pass
            case 'oneway_bf':
                if self.hypothesis == "PAIRWISE":
                    # Use all methods
                    T_p_value = pd.DataFrame(
                        p_value_matrix,
                        columns=self._methods.anova_methods_used
                    )
                    self._tables.oneway_bf = pd.concat([T_hypothesis, T_p_value], axis=1)

                elif self.hypothesis == "FAMILY":
                    # Keep the full table
                    if not isinstance(self._tables.oneway_bf, pd.DataFrame):
                        raise ValueError('oneway_bf should be a pandas dataframe')
                    
                    self._tables.oneway_bf = self._tables.oneway_bf.copy()

                    self._tables.oneway_bf["P-Value"] = p_value_matrix.flatten()
                    self._tables.oneway_bf["Test-Statistic"] = test_stat.flatten()

                    signif_results = self._tables.oneway_bf["P-Value"] < self.alpha
                    self._tables.oneway_bf.loc[signif_results, "Verdict"] = ("Reject Null Hypothesis for Alternative Hypothesis")
                    self._tables.oneway_bf.loc[~signif_results, "Verdict"] = ("Fail to Reject Null Hypothesis")
            case 'twoway':
                pass
            case 'twoway_bf':
                pass
     
    def _show_table(self, table_to_show):
        temp_table = table_to_show.copy()

        if self.hypothesis == "PAIRWISE":
            all_var_names = list(temp_table.columns)
            n_items = len(temp_table)

            # Format all columns except the first
            for col_name in all_var_names[1:]:
                temp_table[col_name] = temp_table[col_name].apply(
                    lambda x: f"{x:6.{self._tables.sig_figs}f}"
                )

            # Reorder columns according to self.ANOVA_Methods
            method_order = {method: i for i, method in enumerate(self._methods.anova_methods)}
            reordered_cols = [all_var_names[0]] + sorted(
                all_var_names[1:], key=lambda col: method_order.get(col, float('inf'))
            )
            temp_table = temp_table[reordered_cols]

        else:  # "FAMILY"
            n_items = len(temp_table)

            # Format 'Test-Statistic' column
            temp_table["Test-Statistic"] = temp_table["Test-Statistic"].apply(
                lambda x: f"{x:6.{self._tables.sig_figs}f}"
            )

            # Format 'P-Value' column
            temp_table["P-Value"] = temp_table["P-Value"].apply(
                lambda x: f"{x:6.{self._tables.sig_figs}f}"
            )

            # Sort rows based on order in self.ANOVA_Methods
            method_order = {method: i for i, method in enumerate(self._methods.anova_methods)}
            temp_table = temp_table.sort_values(
                by=temp_table.columns[0],  # first column, e.g. "Family-Wise Method"
                key=lambda col: col.map(method_order)
            ).reset_index(drop=True)
        
        print(temp_table)
        
    def _data_summary_report_one_way(self, ANOVA_TYPE):
        n_groups = len(self.n_i)

        Mystring = f"\nOne-Way {ANOVA_TYPE} F-ANOVA Data Summary:\n\n"
        Mystring += f"Confidence Level = {(1 - self.alpha) * 100:.3f} %\n"
        Mystring += f"Number of Observations Total = {self.N}\n"
        Mystring += f"Number of Points in Domain = {self.n_domain_points}\n"
        Mystring += f"Number of Groups = {n_groups}\n"
        Mystring += f"Domain Range = [{self.d_grid[0]:.3f}, {self.d_grid[-1]:.3f}]\n"
        Mystring += f"Domain Subset = [{self.grid_bounds[0]:.3f}, {self.grid_bounds[-1]:.3f}]\n"
        Mystring += f"Group Observation Size: [{', '.join(str(x) for x in self.n_i)}]\n"

        if self._labels.group:
            Mystring += f"Group Labels: [{', '.join(self._labels.group)}]\n"

        Mystring += "\n"
        print(Mystring) 
     
    def _data_summary_report_two_way(self, ANOVA_TYPE):
        # Build Secondary Factor Observation Size string (B_obs_string)
        if self.n_ii is None:
            raise ValueError('n_ii attribute is None when it should be populated for this report')
        
        B_obs_string_parts = []
        for k in range(self._groups.A):
            obs = ' '.join(str(x) for x in self.n_ii[k])
            B_obs_string_parts.append(f"[{obs}]")
        B_obs_string = ', '.join(B_obs_string_parts)

        # Start building the summary string
        Mystring = f"\nTwo-Way {ANOVA_TYPE} F-ANOVA Data Summary:\n\n"
        Mystring += f"Confidence Level = {(1 - self.alpha) * 100:.3f} %\n"
        Mystring += f"Number of Observations Total = {self.N}\n"
        Mystring += f"Number of Points in Domain = {self.n_domain_points}\n"
        Mystring += f"Number of Groups within Primary Factor = {self._groups.A}\n"
        Mystring += f"Number of Groups within Secondary Factor= {self._groups.B}\n"
        Mystring += f"Number of Total Groups = {self._groups.AB}\n"
        Mystring += f"Domain Range = [{self.d_grid[0]:.3f}, {self.d_grid[-1]:.3f}]\n"
        Mystring += f"Domain Subset = [{self.grid_bounds[0]:.3f}, {self.grid_bounds[-1]:.3f}]\n"
        Mystring += f"Primary Factor Observation Size: [{', '.join(str(x) for x in self.n_i)}]\n"
        Mystring += f"Secondary Factor Observation Size: [{B_obs_string}]\n"

        if self._labels.primary:
            Mystring += f"Primary Factor Labels: [{', '.join(self._labels.primary)}]\n"

        if self._labels.secondary:
            Mystring += f"Secondary Factor Labels: [{', '.join(self._labels.secondary)}]\n"

        Mystring += "\n"
        print(Mystring)
            
    def _cast_anova_methods(self, method):
        """
        Filters self._methods.anova_methods_used to keep only valid methods.
        Warns about any unrecognized method names.
        """
        valid_methods = set(self._methods.anova_methods)
        original = sorted(list(method), reverse=True) # Maintain L2 first than F-test

        # Keep only valid methods (case-insensitive match)
        filtered = [m for m in original if m in valid_methods]
        excluded = [m for m in original if m not in valid_methods]

        self._methods.anova_methods_used = tuple(filtered)

        if excluded:
            warnings.warn(f"These unknown methods were excluded: {' & '.join(excluded)}")

        assert len(filtered) >= 1, (
            "No ANOVA methods were selected!\n"
            f"Must be at least one of the following: {', '.join(self._methods.anova_methods)}"
        )    
    
    def _validate_instantiation_inputs(self):
        
        # Validate bounds
        if not isinstance(self.grid_bounds, tuple) or len(self.grid_bounds) != 2:
            raise ValueError(f"F-ANOVA bounds must be a tuple of length 2, but got {type(self.grid_bounds).__name__} with value {self.grid_bounds}")
        if not all(isinstance(x, (int, float)) for x in self.grid_bounds):
            raise ValueError(f"F-ANOVA bounds must contain numeric values, but got {self.grid_bounds}")
        
        # Validate d_grid
        self.d_grid = self._cast_to_1D(self.d_grid)

            
        #TODO need to validate more inputs 
        
    def _validate_stat_inputs(self, alpha, n_boot, n_simul, methods, hypothesis):
        # Validate alpha
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        
        self.alpha = alpha
        
        # Validate n_boot and n_simul
        if not isinstance(n_boot, int) or n_boot <= 0:
            raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
        if not isinstance(n_simul, int) or n_simul <= 0:
            raise ValueError(f"n_simul must be a positive integer, got {n_simul}")
        
        self.n_boot = n_boot
        self.n_simul = n_simul
        
        # Validate ANOVA methods
        if methods is not None:
            upper_cased_methods = tuple(s.upper() for s in self._methods.anova_methods)
            for m in methods:
                if m.upper() not in upper_cased_methods:
                    raise ValueError(f"Invalid method: {m}. Must be one of {self._methods.anova_methods}")
                
            self._cast_anova_methods(methods)
        else:
            self._methods.anova_methods_used = self._methods.anova_methods
        
        # Validate hypothesis: Family or Pairwise
        if hypothesis is not None:
            if not isinstance(hypothesis, str):
                raise TypeError(f"'hypothesis' must be a string, got {type(hypothesis).__name__}")

            if not hypothesis.strip():
                raise ValueError("hypothesis string must not be empty or whitespace only")

            if hypothesis.upper() not in self._labels.H0_OneWay:
                raise ValueError(
                    f"Invalid hypothesis: {hypothesis}. Must be one of {self._labels.H0_OneWay}"
                )
            
            self.hypothesis = hypothesis.upper()
            
    def _function_subsetter(self):

        lb = np.min(self.grid_bounds)
        ub = np.max(self.grid_bounds)


        if lb > np.max(self.d_grid):
            raise ValueError(f'The lower bound (lb={lb:g}) is greater than the maximum value in self.d_grid ({np.max(self.d_grid):g}).')


        if ub < np.min(self.d_grid):
            raise ValueError(f'The upper bound (ub={ub:g}) is less than the minimum value in self.d_grid ({np.min(self.d_grid):g}).')

        subset_idx = np.where((self.d_grid >= lb) & (self.d_grid <= ub))[0]

        subset_d_grid = self.d_grid[subset_idx]

        local_min_idx = np.argmin(subset_d_grid)
        local_max_idx = np.argmax(subset_d_grid)

        global_min_idx = subset_idx[local_min_idx]
        global_max_idx = subset_idx[local_max_idx]

        self.lb_index = global_min_idx
        self.ub_index = global_max_idx

    def _verifyIndicator(self):
        """
        Validates and standardizes subgroup_indicator.
        Accepts either:
        - A list of 1D NumPy arrays (like a cell array in MATLAB)
        - A single 1D or 2D NumPy array
        Ensures final result is a 1D NumPy array of length N.
        """
        
        # Flatten (from list or array) then reshape once
        if isinstance(self._groups.subgroup_indicator, list):
            assert len(self._groups.subgroup_indicator) == self._groups.k, (
                f"Expected {self._groups.k} subgroups, got {len(self._groups.subgroup_indicator)}"
            )
            flat = np.concatenate(self._groups.subgroup_indicator, axis=0)
        elif isinstance(self._groups.subgroup_indicator, np.ndarray):
            flat = self._groups.subgroup_indicator
        else:
            raise TypeError("subgroup_indicator must be a list of np.ndarrays or a np.ndarray")

        # Enforce 1D shape
        self._groups.subgroup_indicator = self._cast_to_1D(flat)

        # Final check
        N_subgroup = self._groups.subgroup_indicator.size
        assert N_subgroup == self.N, (f"subgroup_indicator has {N_subgroup} elements, expected {self.N}")

    def _set_up_two_way(self):
        
      
        self._verifyIndicator()
        
        subgroup = cast(np.ndarray, self._groups.subgroup_indicator)  #  it's only for static type checking.
        n_unique_indicators = len(np.unique(subgroup))
        
        self._groups.B = n_unique_indicators


        self._groups.A = self._groups.k
        self._groups.AB = self._groups.A * self._groups.B

        if not self._labels.secondary and not self._labels.primary:
            self.generic_group_labels = True

        if not self._labels.primary:
            self._labels.primary = [f"{i+1}" for i in range(self._groups.k)] # Automatically Assign Primary Numeric Labels (1,2,3,...)

        if not self._labels.secondary:
             self._labels.secondary = [chr(65 + i) for i in range(n_unique_indicators)] # Automatically Assign Alphabetical Labels (A,B,C,...)

        assert len(self._labels.primary) == self._groups.A, "Labels for each Primary factor level must have a one-to-one correspondence to each level"

        assert len(self._labels.secondary) == self._groups.B, "Labels for each Secondary factor level must have a one-to-one correspondence to each level"

        if self.labels.group:
            raise ValueError('TwoWay ANOVA requires using "primary_labels" and "secondary_labels" as input arguments.\nIt doesnt support the "group_labels" argument due to ambiguity.')

    def _set_up_time_bar(self, method):
        match method:
            case "L2-Bootstrap":
                match self.hypothesis:
                    case 'FAMILY':
                        desc = 'Calculating, Family-wise, Bootstrap L2 test'
                    case "PAIRWISE":
                        desc = f'Calculating, Pair-wise ({self._labels.hypothesis}), Bootstrap L2 test'
                    case _:
                        desc = f'Calculating, Effect ({self._labels.hypothesis}), Bootstrap L2 test'
            case "F-Bootstrap":
                match self.hypothesis:
                    case 'FAMILY':
                        desc ='Calculating, Family-wise, Bootstrap F-type test'
                    case "PAIRWISE":
                        desc = f'Calculating, Pair-wise ({self._labels.hypothesis}), Bootstrap F-type test'
                    case _:
                        desc = f'Calculating, Effect ({self._labels.hypothesis}), Bootstrap F-type test'
            case _:
                raise ValueError(f'Unsupported method for TQDM: {method}')

        return desc
    
    def _n_ii_generator(self):
        """
        Creates self.n_ii: a list of lists containing sample sizes for each
        (primary, secondary) combination — used for plotting in Two-Way ANOVA.
        """
        
        subgroup = cast(np.ndarray, self._groups.subgroup_indicator) #  it's only for static type checking.
        labels_primary = cast(list, self._labels.primary) #  it's only for static type checking.
        labels_secondary = cast(list, self._labels.secondary) #  it's only for static type checking.
        
        aflag = utils.aflag_maker(self.n_i)
        bflag = subgroup

        aflag_levels = np.unique(aflag)
        bflag_levels = np.unique(bflag)

        p = len(aflag_levels)  # primary levels
        q = len(bflag_levels)  # secondary levels

        p_cell = []
        error_messages = []

        for i in range(p):
            q_cell = [np.nan] * q
            for j in range(q):
                # Find indices where both flags match the (i, j) combination
                ij_flag = (aflag == aflag_levels[i]) & (bflag == bflag_levels[j])
                n_ij = np.sum(ij_flag)

                if n_ij == 0:
                    error_messages.append(
                        f"A missing combination of data occurs for Primary Label: {labels_primary[i]} and Secondary Label: {labels_secondary[j]}"
                    )

                q_cell[j] = n_ij
            p_cell.append(q_cell)

        if error_messages:
            for msg in error_messages:
                warnings.warn(msg)
            raise ValueError("Missing combinations detected. See warnings above.")

        self.n_ii = p_cell
        
    @staticmethod    
    def _cast_to_1D(arr):
        arr = np.asarray(arr)
                
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("d_grid must contain numeric values.")
        
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return np.ravel(arr)  # returns a view if possible
        else:
            raise ValueError(f"Input must be a 1D vector, or 2D row/column vector, but got array with shape {arr.shape}")
    