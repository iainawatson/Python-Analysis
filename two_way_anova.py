# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:02:27 2020

@author: iain
"""

# =============================================================================
# Perform a 2-way ANOVA
# i.e. multiple independent variables, one dependent variable
# 
# runs on specific file titled combined data
#
# Optionally run grubbs test if normality fails
# =============================================================================
import os
import pandas as pd
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from scipy import stats
import pathlib
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import numpy as np
import shutil
import sys

# =============================================================================
# Variables
# =============================================================================
# Enter names as list for renaming dataframe columns, ie the first set of independent variables
groups = ['eGFP', 'WT', 'A30P']
# Enter the second set of independent variables
subgroups = ['Bouton', 'Axon']
# Enter the dependent variable
dep_var = 'intensity'
# Enter the overall name of the analysis
output_name = 'mycbp2_intensity'
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

# Calculating effect size
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'mean_sq', 'df', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

def calculate_effect_size(t, left_n, right_n):
    dof = left_n + right_n - 2
    effect_size_r = np.sqrt(t**2/(t**2 + dof))
    return effect_size_r

# =============================================================================
# Get the long formatted data
# =============================================================================
file_list = os.listdir()
for i in file_list:
    if i.endswith('_combined_data.csv'):
        file = i
    elif i.endswith('_outliers_removed.csv'):
        file = i
df = pd.read_csv(file)

# =============================================================================
# Summary data
# =============================================================================
# Also adds the median values to the summary tables
dep_var_summary = rp.summary_cont(df[dep_var])
temp_df = df[[dep_var]].median()
dep_var_summary['Median'] = temp_df.loc[dep_var,]

group_summary = rp.summary_cont(df.groupby(['group'], sort = False)[dep_var])
temp_df = df.groupby(['group'], as_index = False, sort = False)[dep_var].median()
group_summary['Median'] = temp_df.loc[:, dep_var].values

subgroup_summary = rp.summary_cont(df.groupby(['subgroup'], sort = False)[dep_var])
temp_df = df.groupby(['subgroup'], as_index = False, sort = False)[dep_var].median()
subgroup_summary['Median'] = temp_df.loc[:, dep_var].values

all_groups_summary = rp.summary_cont(df.groupby(['subgroup','group'], sort = False)[dep_var])
temp_df = df.groupby(['subgroup','group'], as_index = False, sort = False)[dep_var].median()
all_groups_summary['Median'] = temp_df.loc[:, dep_var].values

output = output_name + '_twoway_anova_statistics_results.txt'
with open(output, 'w') as f:
    f.write('==================\n')
    f.write('SUMMARY STATISTICS\n')
    f.write('==================\n')
    print(dep_var_summary, file = f)
    f.write('\n')
    print(group_summary, file = f)
    f.write('(Use for main effect summary of group)\n')
    f.write('\n')
    print(subgroup_summary, file = f)
    f.write('(Use for main effect summary of subgroup)\n')
    f.write('\n')
    print(all_groups_summary, file = f)
    f.write('\n')

# =============================================================================
# 2-way ANOVA
# Using Type II sum of squares
# See here for more info: http://md.psych.bio.uni-goettingen.de/mv/unit/lm_cat/lm_cat_unbal_ss_explained.html
# =============================================================================
# Fits the model with the interaction term
# This will also automatically include the main effects for each factor
model = ols(dep_var + ' ~ C(subgroup)*C(group)', df).fit()
# Seeing if the overall model is significant
twoway_anova_summary = (f"Overall model F({model.df_model:.0f},{model.df_resid:.0f})={model.fvalue:.2f}, p={model.f_pvalue:.4f}")

# =============================================================================
# Check assumptions for test
# =============================================================================
model_summary = model.summary()
with open(output, 'a') as f:
    f.write('==================================================\n')
    f.write('DETAILED TWO-WAY ANOVA MODEL AS REGRESSION FORMULA\n')
    f.write('==================================================\n')
    f.write(twoway_anova_summary + '\n')
    print(model_summary, file = f)
    f.write('The Durban-Watson test is to detect the presence of autocorrelation\n')
    f.write('Jarque-Bera tests the assumption of normality (check p value)\n')
    f.write('Omnibus tests the assumption of homogeneity of variance (check p value)\n')
    f.write('Condition Number assess multicollinearity (should be under 20)\n')
    f.write('\n')

# =============================================================================
# Test data for normality using Shapiro-Wilk Test, Q-Q plots and histograms
# =============================================================================
sw_stat = []
sw_pv = []
sw_n = []
for i in subgroups:
    for j in groups:
        df_i = df[df['subgroup'] == i]
        df_ij = df_i[df_i['group']== j]
        n = df_ij.shape[0]
        stat, pv = stats.shapiro(df_ij[dep_var])
        sw_stat.append(stat)
        sw_pv.append(pv)
        sw_n.append(n)
    
sw_result = 'Passed Shapiro-Wilk normality test on all conditions.'    
for i in sw_pv:
    if i < 0.05:
        sw_result = 'FAILED Shapiro-Wilk normality test. See below results, check for outliers, Q-Q plots and histograms.'

with open(output, 'a') as f:
    f.write('==========================================\n')
    f.write('TEST FOR NORMALITY USING SHAPIRO-WILK TEST\n')
    f.write('==========================================\n')
    ticker = 0
    for i in subgroups:
        for j in groups:
            stat = sw_stat[ticker]
            pv = sw_pv[ticker]
            n = sw_n[ticker]
            if pv < 0.05:
                pof = 'FAILED Shapiro-Wilk normality test, data is NOT normally distributed'
            else:
                pof = 'passed Shapiro-Wilk normality test, data is normally distributed'
            print(i + ' ' + j + ' ' + pof + ', W(' + str(n) + ')=' + str(round(stat, 2)) + ', p=' + str(round(pv, 4)), file = f)
            ticker = ticker + 1
    f.write('\n')


cwd = os.getcwd()
# Make Q-Q plots and histograms for the data
qq_dir = cwd + '/qq_plots/'
if not os.path.exists(pathlib.Path(qq_dir)):
    os.makedirs(pathlib.Path(qq_dir))
hist_dir = cwd + '/histograms/'
if not os.path.exists(pathlib.Path(hist_dir)):
    os.makedirs(pathlib.Path(hist_dir))
for i in subgroups:
    for j in groups:
        df_i = df[df['subgroup'] == i]
        df_ij = df_i[df_i['group']== j]
        qqplot(df_ij[dep_var], line='s')
        qq_file = pathlib.Path(qq_dir + i + '_' + j + '_qq_plot.pdf')
        pyplot.savefig(qq_file)
        pyplot.clf()
        pyplot.hist(df_ij[dep_var])
        hist_file = pathlib.Path(hist_dir + i + '_' + j + '_histogram.pdf')
        pyplot.savefig(hist_file)
        pyplot.clf()
    
# =============================================================================
# Print results to console
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
print(text)

# =============================================================================
# Perform the Grubbs test
# =============================================================================
if file.endswith('_outliers_removed.csv'):
    answer = 'n'
else:
    answer = input('Perform Grubb\'s Test if normality failed? (y/n): ')
    print('')
    
if answer == 'y':
    # Calculate the z score for all the data sets. Scores are converted to absolutes
    df_grubbs = pd.DataFrame()
    for i in subgroups:
        for j in groups:
            df_i = df[df['subgroup'] == i]
            df_ij = df_i[df_i['group']== j]
            zscore = abs(stats.zscore(df_ij[dep_var]))
            df_ij = df_ij.copy()
            df_ij['z_score'] = zscore
            max_deviation = max(zscore)
            size = df_ij.shape[0]
            crit_z = calculate_critical_value(size, alpha)
            outlier = crit_z - max_deviation
            df_ij = df_ij.copy()
            df_ij['critical_z'] = crit_z
            df_ij.loc[df_ij['z_score'] > crit_z, 'Significant outlier?'] = 'True'
            df_ij.loc[df_ij['z_score'] <= crit_z, 'Significant outlier?'] = 'False'
            df_grubbs = df_grubbs.append(df_ij)
    # Remove the outliers
    df_outliers_removed = df_grubbs[df_grubbs['Significant outlier?'] == 'False']
    original_titles = list(df)
    df_outliers_removed = df_outliers_removed.loc[:, original_titles]
    
    # Output the Grubbs test result
    grubbs_dir = cwd + '/grubbs_test/'
    if not os.path.exists(pathlib.Path(grubbs_dir)):
        os.makedirs(pathlib.Path(grubbs_dir))
    path2 = pathlib.Path(grubbs_dir + output_name + '_grubbs_test_result.csv')
    df_grubbs.to_csv(path2, index=False)
    print('===================')
    print('GRUBBS TEST RESULTS')
    print('===================')
    print(df_grubbs)
    
    # Output csv with outliers removed
    path3 = pathlib.Path(grubbs_dir + output_name + '_outliers_removed.csv')
    df_outliers_removed.to_csv(path3, index=False)
    
    print('\nRerun Two-Way ANOVA on data without outliers')
    for i in file_list:
        if i.endswith('two_way_anova.py'):
            py_file = i
    shutil.copy(py_file, grubbs_dir)
    input('Press any key to exit')
    sys.exit()

# =============================================================================
# Create the ANOVA table and report interaction
# =============================================================================
res = sm.stats.anova_lm(model, typ= 2)
interaction_term = res.iloc[2,3]

int_term_fstat = round(res.iloc[2,2], 2)
int_term_pval = round(interaction_term, 4)
int_term_dof = round(res.iloc[2,1], 0)
residual = round(res.iloc[3,1], 0)
interaction_result = 'A two-way ANOVA was conducted that examined the effect of the factors subgroup and group on ' + dep_var + '.\nThere was a statistically significant interaction between the effects of subgroup and group on ' + dep_var + ', F(' + str(int_term_dof) + ',' + str(residual) + ')=' + str(int_term_fstat) + ', p=' + str(int_term_pval)
if interaction_term > 0.05:
    interaction_result = 'A two-way ANOVA was conducted that examined the effect of the factors subgroup and group on ' + dep_var + '.\nThere was NOT a statistically significant interaction between the effects of subgroup and group on ' + dep_var + ', F(' + str(int_term_dof) + ',' + str(residual) + ')=' + str(int_term_fstat) + ', p=' + str(int_term_pval)

res_with_effect_size = anova_table(res)

with open(output, 'a') as f:
    f.write('================================================\n')
    f.write('TWO-WAY ANOVA TABLE USING TYPE II SUM OF SQUARES\n')
    f.write('================================================\n')
    print(interaction_result, file = f)
    f.write('\n')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res_with_effect_size, file = f)
    f.write('Intepretation of effect size eta\u00b2: 0.01 ~ small, 0.06 ~ medium, >0.014 ~ large\n')
    f.write('\n')

# =============================================================================
if interaction_term > 0.05:
    # =============================================================================
    # Main effects of each factor independently
    # =============================================================================
    # Fits the model
    model2 = ols(dep_var + ' ~ C(subgroup)+ C(group)', df).fit()
    twoway_anova_summary2 = (f"Overall model F({model2.df_model:.0f},{model2.df_resid:.0f})={model2.fvalue:.2f}, p={model2.f_pvalue:.4f}")
    
    # =============================================================================
    # Check assumptions for test
    # =============================================================================
    model_summary2 = model2.summary()
    with open(output, 'a') as f:
        f.write('=======================================================\n')
        f.write('DETAILED TWO-WAY ANOVA MODEL AS REGRESSION FORMULA\n')
        f.write('MAIN EFFECTS FOR EACH FACTOR INDEPENDENT OF INTERACTION\n')
        f.write('=======================================================\n')
        f.write(twoway_anova_summary2 + '\n')
        print(model_summary2, file = f)
        f.write('The Durban-Watson test is to detect the presence of autocorrelation\n')
        f.write('Jarque-Bera tests the assumption of normality (check p value)\n')
        f.write('Omnibus tests the assumption of homogeneity of variance (check p value)\n')
        f.write('Condition Number assess multicollinearity (should be under 20)\n')
        f.write('\n')
    
    # =============================================================================
    # Print model2 results to console
    # =============================================================================
    with open(output, 'r') as f:
        text = f.read()
    print(text)
    
    # =============================================================================
    # Create the ANOVA table and report main effects
    # =============================================================================
    res2 = sm.stats.anova_lm(model2, typ= 2)
    res_with_effect_size2 = anova_table(res2)
    residual2 = round(res2.iloc[2,1], 0)
    
    subgroup_fstat = round(res2.iloc[0,2], 2)
    subgroup_pval = round(res2.iloc[0,3], 4)
    subgroup_dof = round(res2.iloc[0,1], 0)
    subgroup_result = 'The subgroup factor has an independent effect on ' + dep_var + '.\nThere was a statistically significant effect of subgroup on ' + dep_var + ', F(' + str(subgroup_dof) + ',' + str(residual2) + ')=' + str(subgroup_fstat) + ', p=' + str(subgroup_pval)
    if res2.iloc[0,3] > 0.05:
        subgroup_result = 'The subgroup factor DOES NOT have an independent effect on ' + dep_var + '.\nThere was NOT a statistically significant effect of subgroup on ' + dep_var + ', F(' + str(subgroup_dof) + ',' + str(residual2) + ')=' + str(subgroup_fstat) + ', p=' + str(subgroup_pval)
    
    group_fstat = round(res2.iloc[1,2], 2)
    group_pval = round(res2.iloc[1,3], 4)
    group_dof = round(res2.iloc[1,1], 0)
    group_result = 'The group factor has an independent effect on ' + dep_var + '.\nThere was a statistically significant effect of group on ' + dep_var + ', F(' + str(group_dof) + ',' + str(residual2) + ')=' + str(group_fstat) + ', p=' + str(group_pval)
    if res2.iloc[1,3] > 0.05:
        group_result = 'The group factor DOES NOT have an independent effect on ' + dep_var + '.\nThere was NOT a statistically significant effect of group on ' + dep_var + ', F(' + str(group_dof) + ',' + str(residual2) + ')=' + str(group_fstat) + ', p=' + str(group_pval)
    
    with open(output, 'a') as f:
        f.write('=======================================================\n')
        f.write('TWO-WAY ANOVA TABLE USING TYPE II SUM OF SQUARES\n')
        f.write('MAIN EFFECTS FOR EACH FACTOR INDEPENDENT OF INTERACTION\n')
        f.write('=======================================================\n')
        print('Subgroups = ', subgroups, file = f)
        print(subgroup_result, file = f)
        f.write('\n')
        print('Groups = ', groups, file = f)
        print(group_result, file = f)
        f.write('\n')
        print(res_with_effect_size2, file = f)
        f.write('Intepretation of effect size eta\u00b2: 0.01 ~ small, 0.06 ~ medium, >0.014 ~ large\n')
        f.write('\n')

# =============================================================================
# ONLY RUN POST HOC TESTS IF THE INTERACTION IS INSIGNIFICANT
# =============================================================================
if interaction_term > 0.05:
        
    # =============================================================================
    # Tukey's post-hoc analysis
    # =============================================================================
    combined_df = pd.DataFrame()
    for i in subgroups:
        for j in groups:
            df_i = df[df['subgroup'] == i]
            df_ij = df_i[df_i['group']== j]
            tempdf = pd.DataFrame({'subgroup + group':i + '_' + j, dep_var:df_ij[dep_var]})
            combined_df = combined_df.append(tempdf)
    
    # Using Tukey's HSD method. Can use Bonferonni correction to determine effect size
    mc = MultiComparison(combined_df[dep_var], combined_df['subgroup + group'])
    mc_results = mc.tukeyhsd()
    
    if True in mc_results.reject:
        sig_hsd = True
    else:
        sig_hsd = False
    
    with open(output, 'a') as f:
        f.write('==============================\n')
        f.write('POST-HOC COMPARISONS TUKEY HSD\n')
        f.write('==============================\n')
        f.write('All levels compared:')
        print(mc_results, file = f)
        f.write('\n')
    
    # =============================================================================
    # Bonferroni Post Hoc Test
    # =============================================================================
    ttest_group_array = combined_df['subgroup + group'].value_counts()
    ttest_groups = ttest_group_array.index.tolist()
    len_test_groups = len(ttest_groups)
    no_of_comparisons = len_test_groups*(len_test_groups-1)/2
    bon_cutoff = 0.05/no_of_comparisons
    
    bonferroni_table = pd.DataFrame(columns = ['group1', 'group2', 'dof', 't_value', 'p_value', 'significant', 'effect_size_r'])
    comparison_check_list = []
    for x, left in enumerate(ttest_groups):
        for y, right in enumerate(ttest_groups):
            new_check = [left, right]
            comparison_check_list.append(new_check)
            if not x == y and ([left, right] not in comparison_check_list or [right, left] not in comparison_check_list):
                left_n = ttest_group_array[left]
                right_n = ttest_group_array[right]
                tstat, pval = stats.ttest_ind(combined_df[dep_var][combined_df['subgroup + group'] == left], combined_df[dep_var][combined_df['subgroup + group'] == right])
                sig = 'False'
                effect_size = ''
                if pval < bon_cutoff:
                    sig = 'True'
                    effect_size = calculate_effect_size(tstat, left_n, right_n)
                dof = left_n + right_n - 2
                new_row = {'group1':left, 'group2':right, 'dof':dof, 't_value':abs(round(tstat, 2)), 'p_value':round(pval, 4), 'significant':sig, 'effect_size_r':effect_size}
                bonferroni_table = bonferroni_table.append(new_row, ignore_index = True)   
    
    sig_bon = True in bonferroni_table['significant']
    
    with open(output, 'a') as f:
        f.write('==========================================================\n')
        f.write('BONFERRONI CORRECTION POST-HOC COMPARISONS AND EFFECT SIZE\n')
        f.write('==========================================================\n')
        f.write('All levels compared, corrected p=' + str(bon_cutoff) + '\n')
        print(bonferroni_table, file = f)
        f.write('\n')
            
# =============================================================================
# Overeall summary
# =============================================================================
with open(output, 'r') as f:
    text = f.read()
with open(output, 'w') as f:
    f.write('=====================\n')
    f.write('TWO-WAY ANOVA SUMMARY\n')
    f.write('=====================\n')
    print(interaction_result, file = f)
    f.write('\n')
    if interaction_term > 0.05:
        print('Subgroups = ', subgroups, file = f)
        print(subgroup_result, file = f)
        f.write('\n')
        print('Groups = ', groups, file = f)
        print(group_result, file = f)
        f.write('\n')
        if sig_hsd == True:
            f.write('Report Mean and Std Dev for group comparisons\ne.g. Post hoc comparisons using the Tukey HSD test indicated that the mean score for the x condition (M=?, SD=?) was significantly different than the y condition (M=?, SD=?)\n')
            print(mc_results, file = f)
        f.write('\n')
        if sig_bon == True:
            f.write('Planned post-hoc testing, using the Bonferroni correction showed that independent varibale 1 significantly incresed the dependent variable compared to independent variable 2\nt(?)=?, p<'+str(round(bon_cutoff, 4))+', r=?\n')
            print(bonferroni_table, file = f)
    f.write('\n')
    f.write(sw_result + '\n')
    f.write('\n')
    f.write(text)

# =============================================================================
# Overeall summary
# =============================================================================
