import pandas as pd
import numpy as np


years = ["21-22", "22-23", "23-24"]


# Used below for loop to create player data files for each year 
"""
# For each year make a combined csv file with all advanced analytics 
for year in years:
    df_ind = pd.read_csv(f"Player Season Totals (Individual) {year} - Natural Stat Trick.csv")
    df_On_I = pd.read_csv(f"Player Season Totals (On-Ice) {year} - Natural Stat Trick.csv")
    
    # Merge the dataframes
    df_player_data = pd.merge(df_ind, df_On_I)
    
    # Save the merged dataframe to a new CSV file
    df_player_data.to_csv(f"{year}_PlayerData.csv", index=False)
    
"""

# print out dimensions to see if all_player_data has all rows and columns
df1 = pd.read_csv("21-22_PlayerData.csv")
print(df1.shape)
df2 = pd.read_csv("22-23_PlayerData.csv")
print(df2.shape)
df3 = pd.read_csv("23-24_PlayerData.csv")
print(df3.shape)

all_player_data = pd.concat([df1, df2, df3])

print(all_player_data.shape)

all_player_data.to_csv("All_Player_Data.csv", index=False)

    


