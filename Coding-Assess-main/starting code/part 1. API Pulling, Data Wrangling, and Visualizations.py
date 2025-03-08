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


fred = Fred(api_key='60af7aa584e88ed55ad85bad09c2c9d7')