# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 08:45:01 2020

@author: iain
"""

# =============================================================================
# Perform a Grubbs test
# =============================================================================

import os
import pandas as pd
import pathlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from scipy import stats
import numpy as np

# =============================================================================
# Variables
# =============================================================================
# significant cutoff for Grubbs test
alpha = 0.05

# =============================================================================
# Functions
# =============================================================================
def calculate_critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    return critical_value

# =============================================================================
# Get .csv file and output directory
# =============================================================================
cwd = os.getcwd()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv to analyse with Grubbs test')
path = Path(file_path)
file = path.name
df1 = pd.read_csv(file)
output_path = file_path.strip(file)
grubbs_dir = cwd + '/grubbs_test/'
output_file = grubbs_dir + file.strip('.csv') + '_grubbs_test_results.txt'
if not os.path.exists(pathlib.Path(grubbs_dir)):
    os.makedirs(pathlib.Path(grubbs_dir))

# =============================================================================
# Generate individual dataframes for the data
# =============================================================================
    
groups = list(df1)
df_dropna_ls = []
for i in range(len(groups)):
    group = groups[i]
    df_dropna_ls.append(df1[[group]].dropna())
    
# =============================================================================
# Perform the Grubbs test
# =============================================================================
# Calculate the z score for all the data sets. Scores are converted to absolutes
df_zscores = []
outliers = []
critical_zs = {}
for i in range(len(groups)):
    group = groups[i]
    df = df_dropna_ls[i]
    zscore = abs(stats.zscore(df[group]))
    df['z_score'] = zscore
    max_deviation = max(zscore)
    size = df.shape[0]
    crit_z = calculate_critical_value(size, alpha)
    outlier = crit_z - max_deviation
    df.loc[df['z_score'] > crit_z, 'Significant outlier?'] = 'True'
    df.loc[df['z_score'] <= crit_z, 'Significant outlier?'] = 'False'
    if outlier < 0:
        ind = df.loc[df['z_score'] == max_deviation].index[0]
        df1.at[ind, group] = np.NaN
    critical_zs[group] = crit_z
    df_zscores.append(df)

# =============================================================================
# Output the Grubbs test result
# =============================================================================
if os.path.exists(pathlib.Path(output_file)):
    os.remove(pathlib.Path(output_file))
with open(output_file, 'a') as f:
    f.write('GRUBBS TEST RESULTS\n')
    f.write('===================\n')
    for i in range(len(groups)):
        print(groups[i], file = f)
        print(df_zscores[i], file = f)
        print('Critical value of Z: ' + str(round(critical_zs[groups[i]], 6)), file = f)
        f.write('\n')

# =============================================================================
# Output csv with outliers removed
# =============================================================================
path = pathlib.Path(grubbs_dir + file.strip('.csv') + '_outliers_removed.csv')
df1.to_csv(path, index=False)
