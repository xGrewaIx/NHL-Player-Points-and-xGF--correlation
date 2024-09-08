import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

data = pd.read_csv("All_Player_Data.csv")


# At the end find correlation between all statistics 
# Only choose the numeric data types
stats_only_df = data.select_dtypes(include=[np.number])
corr_matrix = stats_only_df.corr()


# Extracting values from the correlation matrix that are greater than 0.65
# using stack() to reshape df by converting column label to a row index
# source: https://sparkbyexamples.com/pandas/pandas-stack-function/#:~:text=Pandas%20stack()%20Usage&text=stack()%20function%20reshapes%20the,row%20level%20to%20column%20level.
# Also I will reset the indexs
# source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html 
high_corr = corr_matrix[corr_matrix > 0.65].stack().reset_index()

# Remove self-correlation
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]  

# rename columns 
high_corr.columns = ['Attribute 1', 'Attribute 2', 'Correlation']

# sort by correlation 
high_corr = high_corr.sort_values(by='Correlation')


# Comment out the making of the csv as already made
# high_corr.to_csv("HighCorr.csv")
# corr_matrix.to_csv("CorrelationMatrix.csv")

"""
The following attributes have a correlation of .65 or more with Total Points:
- (Not going to use Total assists, First assists, Goals, GF)
- iSCF, Shots, iFF, ixG, Off.Â Zone Starts, iCF, Takeaways, iHDCF

The following attributes have a correlation of .65 or more with xGF%:
- SCF%, HDCF%, FF%, SF% (% of total shots for their team), CF%, MDCF%, 
    LDCF%
    
Utilzing the above Create visualizations to see these correlations
"""

corr_TP_col = ['iSCF', 'Shots', 'iFF', 'ixG', 'Off. Zone Starts', 'iCF',
               'Takeaways', 'iHDCF']

for col in corr_TP_col:
    
    plt.scatter(data['Total Points'], data[col], color='blue')
    plt.title(f"Total Points vs {col}")
    plt.xlabel("Total Points")
    plt.ylabel(col)
    plt.show()


corr_xGF_col = ['SCF%', 'HDCF%', 'FF%', 'SF%', 'CF%', 'MDCF%', 'LDCF%']
    
for col in corr_xGF_col:
    
    plt.scatter(data['xGF%'], data[col], color='red')
    plt.title(f"xGF% vs {col}")
    plt.xlabel("xGF%")
    plt.ylabel(col)
    plt.show()
    
    
    
"""
The plots showcase what is shown in the correlation matrix. There is a
relationship between Total Points and xGF% with their 
corresponding columns. 
"""