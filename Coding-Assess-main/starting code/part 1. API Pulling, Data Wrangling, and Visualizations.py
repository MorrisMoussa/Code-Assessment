# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:33:20 2025

@author: morri
"""

#key:
#60af7aa584e88ed55ad85bad09c2c9d7

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime as dateime
import plotly.express as px
import tensorflow as tf
import scipy as sp
from scipy import optimize, stats
#imprt sklearn as sp
from sklearn import linear_model
import statsmodels.api as sm
import os

wd = os.getcwd()
data_directory = str(wd) + "\\Coding-Assess-main\\data\\"
input_file = "Part 1. bonds_yields.xlsx"
df_input pd.read_excel(data_directory + input_file)


def fetch_fred_yield(series_id, start_date, end_date, api_key):
   
    df = pd.DataFrame(fred.get_series(series_id, observation_start=start_date, observation_end= end_date, frequency = 'd')
                      , columns= [series_id])
    df.index.name = 'date'
    return df  

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

df1 = fetch_fred_yield(tenor_series_ids[0], start_date, end_date, my_api_key)








