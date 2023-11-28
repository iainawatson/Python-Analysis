# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:28:07 2020

@author: iain
"""

# %%===========================================================================
# Using to measure one-way ANOVA and report the statistics in a text file
# Performs a Tukey HSD post-Hoc analysis
# =============================================================================

import os
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import pathlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import researchpy as rp

# %%===========================================================================
# Variables
# =============================================================================
#enter the dependent variable for the analysis, eg area, count...
dep_var = 'intensity'

# %%===========================================================================
# Functions
# =============================================================================
# Creates function for ANOVA table to calculate effect size via eta squared and omega squared
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

# remove the last 'n' characters from a string 's'
def remove_end_char(s, n):    
    # Stores the resultant string
    res = ''      
    # Traverse the string
    for i in range(len(s)-n):
        # Insert current character
        res += s[i]  
    # Return the string
    return res

# checks for whether the filename is too long for windows
def check_filename_length(check_name, maximum_length, extension_inc_dot):
    fullname_path = pathlib.Path(check_name)
    str_path = str(fullname_path)
    len_path = len(str_path)
    if len_path > maximum_length:
        diff = len_path - maximum_length
        save_nm = remove_end_char(str_path, diff)
        return save_nm + extension_inc_dot
    else:
        save_nm = str_path
        return save_nm + extension_inc_dot

# %%===========================================================================
# Get .csv file
# =============================================================================

cwd = os.getcwd()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv to analyse with ANOVA')
path = Path(file_path)
file = path.name
df = pd.read_csv(file)
output_nocsv = remove_end_char(file, 4)
test_name = cwd + '/' + output_nocsv + '_statistics_results'
output = check_filename_length(test_name, 255, '.txt')

# %%===========================================================================
# Create summary statistics
# =============================================================================
summary_stats = rp.summary_cont(df)
with open(output, 'w') as f:
    f.write('SUMMARY STATISTICS\n')
    f.write('==================\n')
    print(summary_stats, file = f)
    f.write('\n')
    
# %%===========================================================================
# Create ANOVA table
# =============================================================================
# regroup the analysis into new dataframe of group and result, use dictionary to make dataframe
# technically is carrying out an ordinary least squares regression
groups = list(df)
list_groups = []
list_results = []
for i in groups:  
    for j in range(df.shape[0]):
        list_groups.append(i)
        list_results.append(df.loc[j, i])
newdf = pd.DataFrame({'group':list_groups, dep_var:list_results})
newdf_dropna = newdf.dropna() 
model = ols(dep_var + ' ~ C(group)', data = newdf_dropna).fit()
with open(output, 'a') as f:
    f.write('DETAILED ANOVA MODEL AS REGRESSION FORMULA\n')
    f.write('==========================================\n')
    print(model.summary(), file = f)
    f.write('\n')
    
# %%===========================================================================
# Create an ANOVA table with effect size
# =============================================================================
table = sm.stats.anova_lm(model, typ=2)
table_with_effect_size = anova_table(table)
with open(output, 'a') as f:
    f.write('ANOVA TABLE AND EFFECT SIZE\n')
    f.write('===========================\n')
    print(table_with_effect_size, file = f)
    f.write('\n')

# %%===========================================================================
# Test data for normality
# =============================================================================
# Normality test also in regression table, Jarque-Bera test
# Perfrom Shapiro-Wilk Test good for data sets < 2000 in sample size. Perform for each condition
sw_stat = []
sw_pv = []
sw_n = []
for i in range(len(groups)):
    group = groups[i]
    data1 = df[group]
    data2 = data1.dropna()
    n = data2.shape[0]
    stat, pv = stats.shapiro(data2)
    sw_stat.append(stat)
    sw_pv.append(pv)
    sw_n.append(n)

# Make Q-Q plots for the data
qq_dir = cwd + '/qq_plots/'
if not os.path.exists(pathlib.Path(qq_dir)):
    os.makedirs(pathlib.Path(qq_dir))
for i in range(len(groups)):
    group = groups[i]
    data1 = df[group]
    data2 = data1.dropna()
    qqplot(data2, line='s')
    qq_file = pathlib.Path(qq_dir + group + '_qq_plot.pdf')
    pyplot.savefig(qq_file)
    pyplot.clf()

# Make histograms for the data
hist_dir = cwd + '/histograms/'
if not os.path.exists(pathlib.Path(hist_dir)):
    os.makedirs(pathlib.Path(hist_dir))
for i in range(len(groups)):
    group = groups[i]
    data1 = df[group]
    data2 = data1.dropna()
    pyplot.hist(data2)
    hist_file = pathlib.Path(hist_dir + group + '_histogram.pdf')
    pyplot.savefig(hist_file)
    pyplot.clf()
    
sw_result = 'Passed Shapiro-Wilk normality test on all conditions.'    
for i in sw_pv:
    if i < 0.05:
        sw_result = 'FAILED Shapiro-Wilk normality test. See below results, check for outliers, Q-Q plots and histograms'
        
with open(output, 'a') as f:
    f.write('TEST FOR NORMALITY\n')
    f.write('===================\n')
    for i in range(len(groups)):
        group = groups[i]
        stat = sw_stat[i]
        pv = sw_pv[i]
        n = sw_n[i]
        if pv < 0.05:
            pof = 'FAILED Shapiro-Wilk normality test, data is NOT normally distributed'
        else:
            pof = 'passed Shapiro-Wilk normality test, data is normally distributed'
        print(group + ' ' + pof + ', W(' + str(n) + ')=' + str(round(stat, 2)) + ', p=' + str(round(pv, 4)), file = f)
    f.write('\n')

# %%===========================================================================
# Post-Hoc test for between group differences if ANOVA is significant
# =============================================================================
# Using Tukey's HSD method. Can use Bonferonni correction to determine effect size
mc = MultiComparison(newdf_dropna[dep_var], newdf_dropna['group'])
mc_results = mc.tukeyhsd()
with open(output, 'a') as f:
    f.write('POST-HOC COMPARISONS\n')
    f.write('====================\n')
    print(mc_results, file = f)
    f.write('\n')

# %%===========================================================================
# Overall ANOVA summary
# =============================================================================
Df_model = str(int(model.df_model))
Df_residuals = str(int(model.df_resid))
f_ratio = str(round(model.fvalue, 2))
p_value = str(round(model.f_pvalue, 4))
n = str(len(groups))
if model.f_pvalue < 0.05:
    anova_result = 'ANOVA was significant\nThere was a significant effect of the groups on ' + dep_var +' at the p<.05 level for the ' + n + ' conditions, F(' + Df_model + ',' + Df_residuals + ')=' + f_ratio + ', p=' + p_value
else:
    anova_result = 'ANOVA was not significant\nThere was not a significant effect of the groups on ' + dep_var +' at the p<.05 level for the ' + n + ' conditions, F(' + Df_model + ',' + Df_residuals + ')=' + f_ratio + ', p=' + p_value

with open(output, 'r') as f:
    text = f.read()
with open(output, 'w') as f:
    f.write('ANOVA SUMMARY\n')
    f.write('=============\n')
    f.write(anova_result + '\n')
    f.write('\n')
    if model.f_pvalue < 0.05:
        f.write('Report Mean and Std Dev for group comparisons, see Tukey HSD table below (M=?, SD=?)\ne.g. Post hoc comparisons using the Tukey HSD test indicated that the mean score for the x condition (M=?, SD=?) was significantly different than the y condition (M=?, SD=?)\n')
    f.write(sw_result + '\n')
    f.write('\n')
    f.write(text)

# %%===========================================================================
# Output the rearranged dataframe
# =============================================================================
test_name = cwd + '/' + output_nocsv + '_long_format'
# if the name is too long it overwrites the datafile
if len(test_name) < 259:
    path = check_filename_length(test_name, 255, '.csv')
    newdf_dropna.to_csv(path, index=False)

# %%===========================================================================
# print summary of script
# =============================================================================
print('\nImported file:\n')
print(df)
print('\n------------------------------')
print('\nSummary of results:\n')
with open(output, 'r') as f:
    text = f.read()
    print(text)

