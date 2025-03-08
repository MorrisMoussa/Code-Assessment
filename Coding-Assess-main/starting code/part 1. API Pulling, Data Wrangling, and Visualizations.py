# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:33:20 2025

@author: morris
"""
import pandas as pd 
import numpy as np 
from fredapi import Fred
import os
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

def fetch_fred_yield(series_id, start_date, end_date, api_key):
   
    df = pd.DataFrame(fred.get_series(series_id, observation_start=start_date, observation_end= end_date, frequency = 'd')
                      , columns= [series_id])
    df.index.name = 'date'
    
    return df  

### inputs:
wd = os.getcwd()
data_directory = str(wd) + "\\Coding-Assess-main\\data\\"
input_file = "Part 1. bonds_yields.xlsx"
df_bonds = pd.read_excel(data_directory + input_file)
#my API key
my_api_key = "60af7aa584e88ed55ad85bad09c2c9d7"
# Initialize API key
fred = Fred(api_key=my_api_key)

#date range
start_date = "2023-01-01"
end_date="2023-12-31"
# List of Treasury yield series IDs on FRED
tenor_series_ids = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1",  # Short-term yields
    "DGS2", "DGS3", "DGS5",               # Medium-term yields
    "DGS7", "DGS10", "DGS20", "DGS30"     # Long-term yields
]

#store treasury hisotrical data in a dataframe
for i in range(len(tenor_series_ids)):
    if i == 0:
        df_treasury = fetch_fred_yield(tenor_series_ids[i], start_date, end_date, my_api_key)
    else:
        df_temp = fetch_fred_yield(tenor_series_ids[i], start_date, end_date, my_api_key)
        df_treasury = pd.merge(df_treasury, df_temp,left_index=True, right_index=True, how='left')


df_treasury = df_treasury.rename(columns={   "DGS1" : 1,
    "DGS2" : 2, "DGS3": 3, "DGS5":5,               
    "DGS7":7, "DGS10":10, "DGS20":20, "DGS30":30  
    })

tenors_years = [1,2,3,5,7,10,20,30]


#spread calculations
df_bonds["spread"] = ''
k = 0 
for _, bond in df_bonds.iterrows():
    bond_is = bond['Bond ID']
    wal = float(bond['WAL (years)'])  
    bond_yield = bond['Yield (%)']
    treasury_curve = df_treasury.tail(1)
    yieldd = 0 
    if wal%1 != 0 or wal not in tenors_years: 
        i = 0 
        while tenors_years[i] < wal:
            i = i+1
        low = tenors_years[i-1]
        low_yield = float(treasury_curve[low])
        high = tenors_years[i]
        high_yield = float(treasury_curve[high])
        yieldd = low_yield + (bond_yield-low_yield)*(high_yield - low_yield)/(high - low)
        yieldd = round(yieldd,4)
    
    else:
        yieldd = round(float(treasury_curve[wal]),4)
    df_bonds.iloc[k,4] = bond_yield - yieldd
    k = k+1

# Visualization: Spread Distribution:
plt.figure(figsize=(12, 6))
df_bonds.groupby('Sector')['spread'].mean().sort_values().plot(kind='bar')
plt.ylabel('Average Spread (bps)')
plt.title('Sector-Level Relative Value Analysis')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

df_bonds.boxplot(column='spread', by='Sector', vert=False, grid=False)
plt.xlabel('Spread (bps)')
plt.ylabel('Sector')
plt.title('Spread Distribution by Sector')
plt.suptitle('')  # Remove default Pandas title
plt.show()