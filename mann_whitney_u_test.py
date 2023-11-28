# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:30:41 2020

@author: iain
"""

# =============================================================================
# Perform Mann Whitney U Test
# =============================================================================
# Non-parametric to test two independent groups
# Tests the ranked data after samples failed Shapiro-Wilk test for normality
import os
import tkinter as tk
from tkinter import filedialog
import pathlib
import pandas as pd
import researchpy as rp
from scipy import stats
import numpy as np

# =============================================================================
# Variables
# =============================================================================
#enter the dependent variable for the analysis, eg area, count...
dep_var = 'count'

# =============================================================================
# Get .csv file
# =============================================================================
cwd = os.getcwd()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv to analyse with independent T-test')
path = pathlib.Path(file_path)
file = path.name
df = pd.read_csv(file)
output = file.strip('.csv') + '_mann_whitney_statistics_results.txt'

# =============================================================================
# Create summary statistics
# =============================================================================
# Get median values, required for reporting Mann Whitney
med_values = df.median()
med_df = pd.DataFrame({'median':med_values})
summary_stats = rp.summary_cont(df)
summary_stats['Median'] = med_values.values
with open(output, 'w') as f:
    f.write('SUMMARY STATISTICS\n')
    f.write('==================\n')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary_stats, file = f)
    f.write('\n')

# =============================================================================
# Select columns to analyse
# =============================================================================
groups = list(df)
group_count = len(groups)
if group_count > 2:
    print('Select which columns to compare:')
    for i in range(group_count):
        print(str(i) + ' = ' + groups[i])
    ctrl_cond = int(input('Enter number for control condition: '))
    var_cond = int(input('Enter number for variable condition: '))
else:
    ctrl_cond = 0
    var_cond = 1

# =============================================================================
# Create long format table easier to analyse
# =============================================================================
list_groups = []
list_values = []
for i in groups:
    for j in range(df.shape[0]):
        list_groups.append(i)
        list_values.append(df.loc[j, i])
new_df = pd.DataFrame({'group':list_groups, dep_var:list_values})
new_df.dropna(axis = 0, inplace = True)

# =============================================================================
# Perform Mann Whitney U Test
# =============================================================================
ctrl_data = new_df.loc[new_df['group'] == groups[ctrl_cond]]
var_data = new_df.loc[new_df['group'] == groups[var_cond]]
mw_sv, mw_pv = stats.mannwhitneyu(ctrl_data[dep_var], var_data[dep_var])
# Find the median values
ctrl = groups[ctrl_cond]
var = groups[var_cond]
ctrl_med = str(round(med_df.loc[ctrl,'median'], 2))
var_med = str(round(med_df.loc[var,'median'], 2))
mw_stat = str(round(mw_sv, 2))
p_val = str(round(mw_pv, 4))

if mw_pv < 0.05:
    manntest_result = 'Mann-Whitney U test was significant\nThere was a significant difference in the scores between the control group ' + ctrl + ' (Mdn=' + ctrl_med + ') \nand the independent variable ' +  var + ' (Mdn=' + var_med + ') conditions; U(' + mw_stat + '), p=' + p_val + '.'
else:
    manntest_result = 'Mann-Whitney U test was NOT significant\nThere was no significant difference in the scores between the control group ' + ctrl + ' (Mdn=' + ctrl_med + ') \nand the independent variable ' +  var + ' (Mdn=' + var_med + ') conditions; U(' + mw_stat + '), p=' + p_val + '.'


with open(output, 'r') as f:
    text = f.read()
with open(output, 'w') as f:
    f.write('MANN WHITNEY U SUMMARY\n')
    f.write('==========================\n')
    f.write(manntest_result + '\n')
    f.write('\n')
    f.write(text)

# =============================================================================
# Calculate the effect size eta squared
# =============================================================================
# Calculate the z score from p-value
z_score = stats.norm.ppf(mw_pv)

# Sum the total number of observations in each group
ctrl_n = df[ctrl].count()
var_n = df[var].count()
total_n = ctrl_n + var_n

# Caluclate effect size r
effect_r = z_score/np.sqrt(total_n)

# Caluclate effect size eta squared
effect_eta = z_score**2/total_n

r_effect_interpretation = 'Effect size r intepretaion:\n<0: adverse effect, 0.0<0.1: no effect, 0.1<0.24: small effect, 0.24<0.37: intermediate effect, 0.37<0.45: large effect'
eta_effect_interpretation = 'Effect size eta\u00b2 intepretaion:\n0.00<0.010: no effect, 0.010<0.060: small effect, 0.060<0.140: intermediate effect, 0.140<0.200: large effect'

with open(output, 'a') as f:
    f.write('EFFECT SIZE\n')
    f.write('==================\n')
    f.write('Effect size r:\n' + str(round(effect_r, 2)) + '\n')
    f.write(r_effect_interpretation + '\n')
    f.write('\n')
    f.write('Effect size eta\u00b2:\n' + str(round(effect_eta, 3)) + '\n')
    f.write(eta_effect_interpretation + '\n')
    f.write('\n')

# =============================================================================
# Print results to console
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
print(text + '\n(Results saved locally)')