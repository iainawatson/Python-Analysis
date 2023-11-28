# =============================================================================
# Using to measure Welch T-test and report the statistics in a text file
# =============================================================================
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import pathlib
import researchpy as rp
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import pickle

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
file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv to analyse with Welch\'s T-test')
path = pathlib.Path(file_path)
file = path.name
df = pd.read_csv(file)
output = file.strip('.csv') + '_welchs_ttest_statistics_results.txt'

# =============================================================================
# Create summary statistics
# =============================================================================
summary_stats = rp.summary_cont(df)
with open(output, 'w') as f:
    f.write('SUMMARY STATISTICS\n')
    f.write('==================\n')
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
# Test data for normality
# =============================================================================
# Normality test also in regression table, Jarque-Bera test
# Perfrom Shapiro-Wilk Test good for data sets < 2000 in sample size. Perform for each condition
analysis_groups = [groups[ctrl_cond], groups[var_cond]]
sw_stat = []
sw_pv = []
sw_n = []
for i in range(len(analysis_groups)):
    group = analysis_groups[i]
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
for i in range(len(analysis_groups)):
    group = analysis_groups[i]
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
for i in range(len(analysis_groups)):
    group = analysis_groups[i]
    data1 = df[group]
    data2 = data1.dropna()
    pyplot.hist(data2)
    hist_file = pathlib.Path(hist_dir + group + '_histogram.pdf')
    pyplot.savefig(hist_file)
    pyplot.clf()
    
sw_result = 'Passed Shapiro-Wilk normality test.'    
for i in sw_pv:
    if i < 0.05:
        sw_result = 'FAILED Shapiro-Wilk normality test. See below results, check for outliers, Q-Q plots and histograms.\nConsider Mann-Whitney U Test if NOT normally distributed'
        
with open(output, 'a') as f:
    f.write('TEST FOR NORMALITY\n')
    f.write('===================\n')
    for i in range(len(analysis_groups)):
        group = analysis_groups[i]
        stat = sw_stat[i]
        pv = sw_pv[i]
        n = sw_n[i]
        if pv < 0.05:
            pof = 'FAILED Shapiro-Wilk normality test, data is NOT normally distributed'
        else:
            pof = 'passed Shapiro-Wilk normality test, data is normally distributed'
        print(group + ' ' + pof + ': W(' + str(n) + ')=' + str(round(stat, 2)) + ', p=' + str(round(pv, 4)), file = f)
    f.write('\n')

# =============================================================================
# Check homogeneity of variances using Levene\’s test for homogeneity of variance
# =============================================================================
ctrl_data = new_df.loc[new_df['group'] == groups[ctrl_cond]]
var_data = new_df.loc[new_df['group'] == groups[var_cond]]
l_sv, l_pv = stats.levene(ctrl_data[dep_var], var_data[dep_var])

l_result = 'Passed Levene\'s test for homogeneity of variance.'    
if l_pv < 0.05:
    l_result = 'FAILED Levene\'s test for homogeneity of variance. See below results.\nConsider Welch\'s T-test if normally distributed'

dof = ctrl_data.shape[0] + var_data.shape[0] - len(analysis_groups)

with open(output, 'a') as f:
    f.write('TEST FOR VARIANCE\n')
    f.write('=================\n')
    if l_pv < 0.05:
        f.write('FAILED Levene\'s test for homogeneity: ')
    else:
        f.write('Passed Levenes\'s test for homogeneity: ')
    f.write('F(1,' + str(dof) + ')=' + str(round(l_sv, 2)) + ', p=' + str(round(l_pv, 4)) + '\n')
    f.write('\n')

# =============================================================================
# Perform Welch\'s T-test
# =============================================================================
t_sv, t_pv = stats.ttest_ind(ctrl_data[dep_var], var_data[dep_var], equal_var = False)
t_sv = abs(t_sv)
descriptives, results = rp.ttest(ctrl_data[dep_var], var_data[dep_var])

with open(output, 'a') as f:
    f.write('WELCH\'S T-TEST\n')
    f.write('==================\n')
    print(descriptives, file = f)
    print('\n', file = f)
    print(results, file = f)
    print('\n', file = f)
    
# =============================================================================
# Summarise T-test and print data
# =============================================================================
ctrl_mean = descriptives.loc[0, 'Mean']
ctrl_sd = descriptives.loc[0, 'SD']
var_mean = descriptives.loc[1, 'Mean']
var_sd = descriptives.loc[1, 'SD']

ctrl = analysis_groups[0]
ctrl_mean_str = str(round(ctrl_mean, 2))
ctrl_sd_str = str(round(ctrl_sd, 2))
var = analysis_groups[1]
var_mean_str = str(round(var_mean, 2))
var_sd_str = str(round(var_sd, 2))
t_sv_str = str(round(t_sv, 2))
t_pv_str = str(round(t_pv, 4))
dof_str = str(dof)

if t_pv < 0.05:
    ttest_result = 'Welch\'s T-test was significant\nThere was a significant difference in the scores between the control group ' + ctrl + ' (M=' + ctrl_mean_str + ', SD=' + ctrl_sd_str + ') \nand the independent variable ' +  var + ' (M=' + var_mean_str + ', SD=' + var_sd_str + ') conditions; t(' + dof_str + ')=' + t_sv_str + ', p=' + t_pv_str + '.'
else:
    ttest_result = 'Welch\'s T-test was NOT significant\nThere was no significant difference in the scores between the control group ' + ctrl + ' (M=' + ctrl_mean_str + ', SD=' + ctrl_sd_str + ') \nand the independent variable ' +  var + ' (M=' + var_mean_str + ', SD=' + var_sd_str + ') conditions; t(' + dof_str + ')=' + t_sv_str + ', p=' + t_pv_str + '.'

with open(output, 'r') as f:
    text = f.read()
with open(output, 'w') as f:
    f.write('WELCH\'S T-TEST SUMMARY\n')
    f.write('==========================\n')
    f.write(l_result + '\n')
    f.write('\n')
    f.write(sw_result + '\n')
    f.write('\n')
    f.write(ttest_result + '\n')
    f.write('\n')
    f.write(text)

# =============================================================================
# Output the rearranged dataframe
# =============================================================================
list_groups = []
list_results = []
for i in analysis_groups:  
    for j in range(df.shape[0]):
        list_groups.append(i)
        list_results.append(df.loc[j, i])
newdf = pd.DataFrame({'group':list_groups, dep_var:list_results})
newdf_dropna = newdf.dropna() 

path = pathlib.Path(os.getcwd() + '/' + file.strip('.csv') + '_long_format.csv')
newdf_dropna.to_csv(path, index=False)

# =============================================================================
# Output tested conditions to file
# =============================================================================
with open('analysis_groups.pickle', 'wb') as f:
    pickle.dump(analysis_groups, f, pickle.HIGHEST_PROTOCOL)
    
# =============================================================================
# Print results to console
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
print(text + '\n(Results saved locally)')