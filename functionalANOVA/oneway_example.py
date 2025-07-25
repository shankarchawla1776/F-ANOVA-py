import numpy as np
import pandas as pd          
import os
from functionalANOVA.core.fanova import functionalANOVA

# Import Data
script_dir = os.path.dirname(__file__)  # folder containing the script
df = pd.read_csv(os.path.join(script_dir,'Data' ,"gait_data.csv"))  # replace with your actual file path

# Extract all group columns based on column name patterns
group1_cols = [col for col in df.columns if col.startswith("group1")]
group2_cols = [col for col in df.columns if col.startswith("group2")]
group3_cols = [col for col in df.columns if col.startswith("group3")]

# Convert to NumPy arrays
group1 = df[group1_cols].to_numpy()
group2 = df[group2_cols].to_numpy()
group3 = df[group3_cols].to_numpy()

# Final structure
group_arrays = [group1, group2, group3]

# Extract time vector
time = df["t"].to_numpy()

# Bounds on time
bounds = (-np.inf, np.inf)

myANOVA = functionalANOVA(data_list=group_arrays, d_grid=time, grid_bounds=bounds,
                          group_labels=['Group A', 'Group B', 'Group C'])