import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Any, ClassVar, cast
from dataclasses import dataclass, field
import warnings
from functionalANOVA.core import utils
from functionalANOVA.core.plot_means import plot_means

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
        self._validate_inputs()
        
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
            warnings.warn(f'Functional data has a resolution of {self.n_domain_points} elements.\nIt is recommended to have a resolution of at least 1000 elements for the convergence of the F-ANOVA p-values')
            
        # Subset  data
        for k in range(self._groups.k):
            self.data.append(data_list[k][self.lb_index : self.ub_index, :]) # trimming
        
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

    def plot_means(self):
        #TODO Migrate and integrate plotting method here
        pass

    def plot_covariances(self):
        #TODO Migrate and integrate plotting method here
        pass

    def _validate_inputs(self):
        
        # Validate bounds
        if not isinstance(self.grid_bounds, tuple) or len(self.grid_bounds) != 2:
            raise ValueError(f"F-ANOVA bounds must be a tuple of length 2, but got {type(self.grid_bounds).__name__} with value {self.grid_bounds}")
        if not all(isinstance(x, (int, float)) for x in self.grid_bounds):
            raise ValueError(f"F-ANOVA bounds must contain numeric values, but got {self.grid_bounds}")
        
        # Validate d_grid
        self.d_grid = self._cast_to_1D(self.d_grid)

            
        #TODO need to validate more inputs 
        
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
    