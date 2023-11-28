# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:50:37 2020

@author: iain
"""

# =============================================================================
# Creates a long format csv from an input of 2 seperate results tables in short format with
# corresponding biological repeat results
# 
# Also includes code to delete columns of data from the analysis
#
# Use to perform 2-way ANOVA
# =============================================================================
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pathlib

# =============================================================================
# Variables
# =============================================================================
# Enter names as list for renaming dataframe columns, ie the first set of independent variables
groups = ['eGFP', 'WT', 'A30P']
# Enter the second set o findependent variables
subgroups = ['Bouton', 'Axon']
# Enter the dependent variable
dep_var = 'area'
# Enter the overall name of the analysis
output_name = 'mycbp2_puncta_area'

# =============================================================================
# Get multiple .csv files, combine and make long format
# =============================================================================
cwd = os.getcwd()
root = tk.Tk()
root.withdraw()
file_paths = []
paths = []
files = []
dfs = []
for i in range(len(subgroups)):
    file_path = filedialog.askopenfilename(initialdir = cwd, title = 'Choose ' + subgroups[i] +' csv to analyse with ANOVA')
    file_paths.append(file_path)
    path = pathlib.Path(file_path)
    paths.append(pathlib.Path(file_path))
    file = path.name
    files.append(path.name)
    df = pd.read_csv(path)
    dfs.append(df)
# Repeat for creating hues for technical repeats on graph
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename(initialdir = cwd, title = 'Choose csv with technical repeats list')
path2 = pathlib.Path(file_path2)
df2 = pd.read_csv(path2)

# =============================================================================
# Remove columns for dataframes here fi necessary
# =============================================================================
raw_titles = list(df2)
# Choose which columns to keep
raw_titles_to_keep = raw_titles[:]
new_df2 = df2.loc[:, raw_titles_to_keep]
new_dfs = []
# Apply the columns to keep to keep in the data csvs
for i in dfs:
    df = i.loc[:, raw_titles_to_keep]
    new_dfs.append(df)
    
# =============================================================================
# Rename the titles to condition names in variables
# =============================================================================
for i in new_dfs:
    i.columns = groups
new_df2.columns = groups

# =============================================================================
# Create long format table of data
# =============================================================================
subgroup = []
group = []
bio_rep = []
data = []
for i in range(len(subgroups)):
    temp_name = subgroups[i]
    df = new_dfs[i]
    for j in groups:
        for k in range(new_df2.shape[0]):
            rep = new_df2.loc[k, j]
            datum = df.loc[k, j]
            subgroup.append(temp_name)
            bio_rep.append(rep)
            group.append(j)
            data.append(datum)
lf_df = pd.DataFrame({'subgroup':subgroup , 'bio_rep':bio_rep, 'group':group, dep_var:data})
lf_df.dropna(axis = 0, subset = [dep_var], inplace = True)

# =============================================================================
# Output the rearranged dataframe
# =============================================================================
path = pathlib.Path(output_name + '_combined_data.csv')
lf_df.to_csv(path, index=False)
print(lf_df)

# =============================================================================
# Choose to normalise the data
# =============================================================================
option = input('Normalise the data? (y/n): ')
if option == 'y':
    for count, var in enumerate(groups):
        print(count, var)
    norm_cond = int(input('Enter corresponding number for condition to normalise to: '))
    for count, var in enumerate(subgroups):
        print(count, var)
    subgroup_cond = int(input('Enter corresponding number for second factor to normalise to: '))
   
    bioreps_list = lf_df['bio_rep'].value_counts(sort = False).index.tolist()
    
    # Create a normalisation factorial for each biorep in control condition, isolate the normalising condition
    tempdf = lf_df.loc[lf_df['group'] == groups[norm_cond]]
    tempdf2 = tempdf.loc[tempdf['subgroup'] == subgroups[subgroup_cond]]
    # Populate a dicitonary with all the factorials
    factorial_dic = {}
    for i in bioreps_list:
        tempdf3 = tempdf2.loc[tempdf2['bio_rep'] == i]
        mean = tempdf3[dep_var].mean()
        factorial = 1/mean
        factorial_dic[i] = factorial
        
    # Apply the factorial to all the bioreps in the original dataframe
    for i in bioreps_list:
        norm_df = lf_df
        temp_df5 = lf_df.loc[lf_df['bio_rep'] == i]
        fact_df = temp_df5[dep_var].apply(lambda x: x * factorial_dic[i])
        index_to_replace = fact_df.index.tolist()
        values_to_replace = fact_df.tolist()
        norm_df.loc[index_to_replace, dep_var] = values_to_replace

# =============================================================================
# Output the normalised dataframe
# =============================================================================
    path = pathlib.Path(output_name + '_normalised_combined_data.csv')
    norm_df.to_csv(path, index=False)
    print(norm_df)
