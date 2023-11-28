# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:22:57 2020

@author: iain
"""

# =============================================================================
# Performs Kruskal_Wallis H Test
# =============================================================================
import os
import tkinter as tk
from tkinter import filedialog
import pathlib
import researchpy as rp
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# %%
# =============================================================================
# Variables
# =============================================================================
#enter the dependent variable for the analysis, eg area, count...
dep_var = 'count'

# %%
# =============================================================================
# Functions
# =============================================================================
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

# %%
# =============================================================================
# Get .csv file
# =============================================================================
cwd = os.getcwd()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv to analyse with ANOVA')
path = pathlib.Path(file_path)
file = path.name
df = pd.read_csv(file)
output_nocsv = remove_end_char(file, 4)
test_name = cwd + '/' + output_nocsv + '_kruskal_wallis_statistics_results'
output = check_filename_length(test_name, 255, '.txt')

# %%
# =============================================================================
# Create summary statistics
# =============================================================================
summary_stats = rp.summary_cont(df)
# add the median values to the summary stats
med_values = df.median()
med_df = pd.DataFrame({'median':med_values})
summary_stats['Median'] = med_values.values
with open(output, 'w') as f:
    f.write('SUMMARY STATISTICS\n')
    f.write('==================\n')
    print(summary_stats, file = f)
    f.write('\n')

# %%
# =============================================================================
# Create long format dataframe    
# =============================================================================
groups = list(df)
list_groups = []
list_results = []
for i in groups:  
    for j in range(df.shape[0]):
        list_groups.append(i)
        list_results.append(df.loc[j, i])
newdf = pd.DataFrame({'group':list_groups, dep_var:list_results})
newdf_dropna = newdf.dropna() 

# %%    
# =============================================================================
# Perform Kruskal-Wallis test
# =============================================================================
k_hv, k_pv = stats.kruskal(*[group[dep_var].values for name, group in newdf_dropna.groupby('group')])
dof = len(groups) - 1
k_hv_str = str(round(k_hv, 2))
k_pv_str = str(round(k_pv, 4))
dof_str = str(dof)
n = str(len(groups))

if k_pv < 0.05:
    kw_result = 'Kruskal-Wallis H test was significant\nThere was a significant effect of the groups on ' + dep_var +' at the p<.05 level for the ' + n + ' conditions, H(' + dof_str + ')=' + k_hv_str + ', p=' + k_pv_str
else:
    kw_result = 'Kruskal-Wallis H test was NOT significant\nThere was no significant effect of the groups on ' + dep_var +' at the p<.05 level for the ' + n + ' conditions, H(' + dof_str + ')=' + k_hv_str + ', p=' + k_pv_str

data = [['chi-square', round(k_hv, 10)], ['df', dof], ['asymp. sig', round(k_pv, 10)]]
kw_results = pd.DataFrame(data, columns = ['','result'])

# Calculate the rank means
rank = []    
for i in range(newdf_dropna.shape[0]):
    rank.append(i+1)
sorted_df = newdf_dropna.sort_values(by=dep_var)
sorted_df['rank'] = rank
rank_means = []
counts = []
for i in range(len(groups)):
    group = groups[i]
    temp_df = sorted_df['group'] == group
    rank_means.append(sorted_df[temp_df]['rank'].mean())
    counts.append(sorted_df[temp_df]['rank'].count())
ranks_results = pd.DataFrame({'group':groups, 'n':counts, 'rank means':rank_means})

with open(output, 'a') as f:
    f.write('KRUSKAL-WALLIS H TEST STATISTICS\n')
    f.write('================================\n')
    print(kw_results.to_string(index = False), file = f)
    f.write('\n')
    print(ranks_results.to_string(index = False), file = f)
    f.write('\n')

# %%
# =============================================================================
# Perform Dunn's post hoc test to compare groups
# =============================================================================
p_value_df = sp.posthoc_dunn(newdf_dropna, val_col=dep_var, group_col='group')
# Using Dunn's method for adjustment of independent variables
adj_p_value_df = sp.posthoc_dunn(newdf_dropna, val_col=dep_var, group_col='group', p_adjust = 'holm')

with open(output, 'a') as f:
    f.write('POST-HOC COMPARISONS\n')
    f.write('====================\n')
    f.write('p-values:\n')
    print(p_value_df.to_string(), file = f)
    f.write('\n')
    f.write('Adjusted p-values (Bonferroni corrected):\n')
    print(adj_p_value_df.to_string(), file = f)
    f.write('\n')

# %%
# =============================================================================
# Overall Kruskal-Wallis summary
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
with open(output, 'w') as f:
    f.write('KRUSKAL-WALLLIS SUMMARY\n')
    f.write('=======================\n')
    f.write(kw_result + '\n')
    f.write('\n')
    if k_pv < 0.05:
        f.write('Report Ranked Means and p-values for group comparisons, see tables below (M=?, SD=?)\ne.g. Post hoc comparisons using the Bonferroni corrected Dunn\'s test indicated ranked means score for the x condition (M=?, p=?) was significantly different than the y condition (M=?, p=?)\n')
    f.write('\n')
    f.write(text)

# %%  
# =============================================================================
# Print results to console
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
print(text + '\n(Results saved locally)')