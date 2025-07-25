from scipy.io import loadmat
import os
import numpy as np
from functionalANOVA.core.fanova import functionalANOVA

# Import Data
script_dir = os.path.dirname(__file__)  # folder containing the script
matlab_data = loadmat(os.path.join(script_dir,'Data', "example_data.mat"))

# Get data out of .mat file
groups = [matlab_data['TwoWayData'][0, 0], matlab_data['TwoWayData'][0, 1]]
time = matlab_data['timeData']
indicator_list = [matlab_data['IndicatorCell'][0,0], matlab_data['IndicatorCell'][0,1]]

# Bounds on time
bounds = (-np.inf, np.inf)

myANOVA = functionalANOVA(data_list=groups, d_grid=time, grid_bounds=bounds, subgroup_indicator=indicator_list)