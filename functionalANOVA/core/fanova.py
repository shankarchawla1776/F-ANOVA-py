import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Any, ClassVar, cast, Sequence
from dataclasses import dataclass, field
import warnings
from functionalANOVA.core import utils
from functionalANOVA.core.methods import oneway, twoway, plotting

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

@dataclass
class HypothesisInfo:  # Used just for vanilla Oneway
    SSH_t: np.ndarray
    pair_vec: list[str]
    q: int
    n_tests: int
    C: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None

@dataclass
class AnovaStatistics:  # Used just for vanilla Oneway
    T_n: np.ndarray
    F_n: Optional[np.ndarray] = None
    beta_hat: Optional[float] = None
    kappa_hat: Optional[float] = None
    beta_hat_unbias: Optional[float] = None
    kappa_hat_unbias: Optional[float] = None

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
        return  self._groups

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
            self._setup_twoway()  # Creates Indicator Matrices and default Labels
            self._n_ii_generator()  # Creates Secondary Size Array

    def plot_means(self,
                   plot_type: str,
                   subgroup_indicator: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                   observation_size_label: bool = True,
                   group_labels: Optional[List[str]] = None,
                   primary_labels: Optional[List[str]] = None,
                   secondary_labels: Optional[List[str]] = None,
                   x_scale: str = '',
                   y_scale: str = '',
                   domain_units_label: str = '',
                   response_units_label: str = '',
                   data_transparency: float = 0.1,
                   legend_transparency: float = 0.3333,
                   data_line_width: float = 1.75,
                   mean_line_width: float = 5,
                   font_size: int = 18,
                   title_labels: Optional[Any] = None,
                   save_path: str = '',
                   legend_location: str = 'best',
                   num_columns: int = 1,
                   legend_title: str = '',
                   new_colors: Optional[np.ndarray] = None,
                   position: Tuple[int, int, int, int] = (90, 90, 1400, 800)) -> Tuple[Any, Any]:

        return plotting.plot_means(self, plot_type, subgroup_indicator, observation_size_label, group_labels, primary_labels, secondary_labels, x_scale, y_scale, domain_units_label, response_units_label, data_transparency, legend_transparency, data_line_width, mean_line_width, font_size, title_labels, save_path, legend_location, num_columns, legend_title, new_colors, position)

    def plot_covariances(self,
                        plot_type: str,
                        subgroup_indicator: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                        group_labels: Optional[List[str]] = None,
                        primary_labels: Optional[List[str]] = None,
                        secondary_labels: Optional[List[str]] = None,
                        x_scale: str = '',
                        y_scale: str = '',
                        color_scale: str = '',
                        domain_units_label: str = '',
                        response_units_label: str = '',
                        title_labels: Optional[Any] = None,
                        save_path: str = '',
                        position: Tuple[int, int, int, int] = (90, 257, 2000, 800)) -> Any:

        return plotting.plot_covariances(self, plot_type, subgroup_indicator, group_labels, primary_labels, secondary_labels, x_scale, y_scale, color_scale, domain_units_label, response_units_label, title_labels, save_path, position)

    def _run_oneway(self, *args, **kwargs):
        return oneway.run_oneway(self, *args, **kwargs)

    def _run_onewayBF(self, *args, **kwargs):
        return oneway.run_onewayBF(self, *args, **kwargs)

    def _run_twoway(self, *args, **kwargs):
        return twoway.run_twoway(self, *args, **kwargs)

    def _run_twowayBF(self, *args, **kwargs):
        return twoway.run_twowayBF(self, *args, **kwargs)

    def oneway(self,
                  n_boot: int = 10_000,
                  n_simul: int = 10_000,
                  alpha: float = 0.05,
                  methods: Optional[Sequence[str]] = None,
                  hypothesis: Optional[Sequence[str]] = None):
        # Still need to add args of GroupLabels and showSimulPlot

        #Sometype of checking for inputs above
        self._validate_stat_inputs(alpha, n_boot, n_simul, methods, hypothesis)
        n_methods = len(self._methods.anova_methods_used)

        eta_i, eta_grand, build_Covar_star = utils.compute_group_means(self._groups.k, self.n_domain_points, self.data, self.n_i, self.N)
        H0 = self._computeSSH_and_pairs(eta_i, eta_grand)

        if np.all(H0.SSH_t < np.finfo(float).eps):
            warnings.warn("Pointwise between-subject variation is 0. Check for duplicated data matrices.")


        # Compute estimated covariance
        gamma_hat = (1 / (self.N - self._groups.k)) * (build_Covar_star @ build_Covar_star.T)  #changed transpose order

        # Only Positive eigen values
        eig_gamma_hat = np.linalg.eigvalsh(gamma_hat)
        eig_gamma_hat = eig_gamma_hat[eig_gamma_hat > 0]


        # Compute SSE(t)
        SSE_t = (self.N - self._groups.k) * np.diag(gamma_hat) # Pointwise within-subject  (Group/Categorical) variations

        match self.hypothesis:
            case 'FAMILY':
                 self._tables.oneway = pd.DataFrame({
                     'Family-Wise Method': list(self._methods.anova_methods_used),
                     'Test-Statistic': [np.nan] * n_methods,
                     'P-Value': [np.nan] * n_methods,
                     'Verdict': [''] * n_methods,
                     'Parameter 1 Name': [''] * n_methods,
                     'Parameter 1 Value': [None] * n_methods,
                     'Parameter 2 Name': [''] * n_methods,
                     'Parameter 2 Value': [None] * n_methods})

            case 'PAIRWISE':
                self._tables.oneway = pd.DataFrame({'Hypothesis': H0.pair_vec})
            case _:
                pass

        T_hypothesis = pd.DataFrame({'Hypothesis': H0.pair_vec})
        params = self._setup_oneway(H0, SSE_t, gamma_hat)

        p_value_matrix = self._run_oneway(eig_gamma_hat, eta_i, params, H0)

        self._prep_tables('oneway', p_value_matrix, T_hypothesis=T_hypothesis, test_stat=None)

        if self.verbose:
            self._data_summary_report_one_way(ANOVA_TYPE='homoskedastic')
            self._show_table(self._tables.oneway)

    def oneway_bf(self,
                  n_boot: int = 10_000,
                  n_simul: int = 10_000,
                  alpha: float = 0.05,
                  methods: Optional[Sequence[str]] = None,
                  hypothesis: Optional[Sequence[str]] = None):

        # Still need to add args of GroupLabels

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

    def _prep_tables(self, anova_method, p_value_matrix, T_hypothesis, test_stat):
        match anova_method:
            case 'oneway':
                match self.hypothesis:
                    case "PAIRWISE":
                        T_p_value = pd.DataFrame(p_value_matrix, columns=self._methods.anova_methods_used)
                        self._tables.oneway = pd.concat([T_hypothesis, T_p_value], axis=1)

                    case "FAMILY":
                        assert self._tables.oneway is not None, 'Oneway table is not properly setup'

                        self._tables.oneway["P-Value"] = np.array(p_value_matrix).flatten()

                        signif_results = self._tables.oneway["P-Value"] < self.alpha

                        self._tables.oneway.loc[signif_results, "Verdict"] = "Reject Null Hypothesis for Alternative Hypothesis"
                        self._tables.oneway.loc[~signif_results, "Verdict"] = "Fail to Reject Null Hypothesis"

                        # if self.consistantTable:
                        #     self.OneWay_P_Table = self.OneWay_P_Table.iloc[:, :4]
            case 'oneway_bf':
                match self.hypothesis:
                    case "PAIRWISE":
                        # Use all methods
                        T_p_value = pd.DataFrame(
                            p_value_matrix,
                            columns=self._methods.anova_methods_used
                        )
                        self._tables.oneway_bf = pd.concat([T_hypothesis, T_p_value], axis=1)

                    case "FAMILY":
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


            # Conditionally format Parameter 1/2 Value columns
            for col in ["Parameter 1 Value", "Parameter 2 Value"]:
                if col in temp_table.columns:
                    temp_table[col] = temp_table[col].apply(
                        lambda x: f"{x:6.{self._tables.sig_figs}f}" if isinstance(x, float) else x
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

    def _setup_oneway(self, H0, SSE_t, gamma_hat):
        import numpy as np

        is_family = self.hypothesis == "FAMILY"

        # Initialize
        T_n = np.sum(H0.SSH_t, axis=0)

        if T_n.ndim == 0:
            T_n = T_n.reshape(1, 1)

        F_n = np.zeros(1)
        beta_hat = 0
        kappa_hat = 0
        beta_hat_unbias = 0
        kappa_hat_unbias = 0

        if self._tables.oneway is None:
            raise ValueError('One Way Table wasnt set up properly')

        # FAMILY: Fill Test-Statistic for L2
        if is_family and any("L2" in m for m in self._methods.anova_methods_used):
            mask = [("L2" in m) for m in self._methods.anova_methods_used]
            self._tables.oneway.loc[mask, "Test-Statistic"] = T_n

        # F-test methods
        if any("F" in m for m in self._methods.anova_methods_used):
            numerator = T_n / H0.q
            denominator = np.sum(SSE_t) / (self.N - self._groups.k)
            F_n = numerator / denominator

            if is_family:
                mask = [("F" in m) for m in self._methods.anova_methods_used]
                self._tables.oneway.loc[mask, "Test-Statistic"] = F_n

        # Naive Approx
        if any("Naive" in m for m in self._methods.anova_methods_used):
            beta_hat = utils.beta_hat(gamma_hat)
            kappa_hat = utils.kappa_hat(gamma_hat)

            if is_family:
                mask_1 = np.array(["Naive" in m for m in self._methods.anova_methods_used])
                mask_2 = np.array(["F" in m for m in self._methods.anova_methods_used])

                only_naive = mask_1 & ~mask_2
                self._tables.oneway.loc[only_naive, "Parameter 1 Name"] = "beta"
                self._tables.oneway.loc[only_naive, "Parameter 2 Name"] = "d"
                self._tables.oneway.loc[only_naive, "Parameter 1 Value"] = beta_hat
                self._tables.oneway.loc[only_naive, "Parameter 2 Value"] = H0.q * kappa_hat

                naive_f = mask_1 & mask_2
                d1 = H0.q * kappa_hat
                d2 = (self.N - self._groups.k) * kappa_hat
                self._tables.oneway.loc[naive_f, "Parameter 1 Name"] = "d1"
                self._tables.oneway.loc[naive_f, "Parameter 2 Name"] = "d2"
                self._tables.oneway.loc[naive_f, "Parameter 1 Value"] = d1
                self._tables.oneway.loc[naive_f, "Parameter 2 Value"] = d2

        # BiasReduced Approx
        if any("BiasReduced" in m for m in self._methods.anova_methods_used):
            beta_hat_unbias = utils.beta_hat_unbias(self.N, self._groups.k, gamma_hat)
            kappa_hat_unbias = utils.kappa_hat_unbias(self.N, self._groups.k, gamma_hat)

            if is_family:
                mask_1 = np.array(["BiasReduced" in m for m in self._methods.anova_methods_used])
                mask_2 = np.array(["F" in m for m in self._methods.anova_methods_used])

                only_bias = mask_1 & ~mask_2
                self._tables.oneway.loc[only_bias, "Parameter 1 Name"] = "beta"
                self._tables.oneway.loc[only_bias, "Parameter 2 Name"] = "d"
                self._tables.oneway.loc[only_bias, "Parameter 1 Value"] = beta_hat_unbias
                self._tables.oneway.loc[only_bias, "Parameter 2 Value"] = H0.q * kappa_hat_unbias

                bias_f = mask_1 & mask_2
                d1 = H0.q * kappa_hat_unbias
                d2 = (self.N - self._groups.k) * kappa_hat_unbias
                self._tables.oneway.loc[bias_f, "Parameter 1 Name"] = "d1"
                self._tables.oneway.loc[bias_f, "Parameter 2 Name"] = "d2"
                self._tables.oneway.loc[bias_f, "Parameter 1 Value"] = d1
                self._tables.oneway.loc[bias_f, "Parameter 2 Value"] = d2

        # Return as dictionary
        return AnovaStatistics(T_n, F_n, beta_hat, kappa_hat, beta_hat_unbias,kappa_hat_unbias)

    def _setup_twoway(self):


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

    def _setup_time_bar(self, method):
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

    def _computeSSH_and_pairs(self, eta_i, eta_grand):
        pair_vec = []

        if self.hypothesis == "FAMILY":
            q = self._groups.k - 1  # rank of contrast matrix
            n_tests = 1
            SSH_k = np.zeros((self.n_domain_points, self._groups.k))

            for k in range(self._groups.k):
                SSH_k[:, k] = self.n_i[k] * (eta_i[:, k] - eta_grand) ** 2

            SSH_t = np.sum(SSH_k, axis=1)
            pair_vec = ["FAMILY"]
            C = None
            D = None

        elif self.hypothesis == "PAIRWISE":
            C = utils.construct_pairwise_contrast_matrix(self._groups.k)  # Expected shape: (n_tests, k_groups)
            n_tests = C.shape[0]

            if self._labels.group is None:
                raise ValueError('Group labels must be set')

            assert len(self._labels.group) == self._groups.k, "Each Group Must Have Exactly One Label Associated to it"

            c = np.zeros((1, self.n_domain_points))
            D = np.diag(1.0 / np.array(self.n_i))
            q = 1  # rank of PAIRWISE contrasts
            SSH_t = np.zeros((self.n_domain_points, n_tests))
            pair_vec = []

            for cc in range(n_tests):
                Ct = C[cc, :]  # shape: (k_groups,)
                part12 = Ct @ eta_i.T - c  # shape: (1, n_domain_points)
                rh_side = Ct @ D @ Ct.T

                if rh_side.ndim == 0:
                    SSH_t[:, cc] = (part12 ** 2).flatten() * 1/rh_side
                else:
                    SSH_t[:, cc] = (part12 ** 2).flatten() * np.linalg.inv(rh_side)



                # Label for the contrast
                indices = np.where(C[cc, :] != 0)[0]
                t1 = self._labels.group[indices[0]]
                t2 = self._labels.group[indices[1]]
                pair_vec.append(f"{t1} & {t2}")
        else:
            raise ValueError(f'Unknown Hypothesis: {self.hypothesis}')

        return HypothesisInfo(SSH_t, pair_vec, q, n_tests, C, D)

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
