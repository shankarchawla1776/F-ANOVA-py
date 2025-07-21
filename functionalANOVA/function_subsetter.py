import numpy as np


def function_subsetter(self):

    lb = np.min(self.bounds_array)
    ub = np.max(self.bounds_array)


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
