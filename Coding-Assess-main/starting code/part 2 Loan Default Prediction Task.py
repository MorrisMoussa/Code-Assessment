# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:22:10 2025

@author: morris
"""
import pandas as pd 
import numpy as np 
from fredapi import Fred
import os
# import tensorflow as tf
# import scipy as sp
# from scipy import optimize, stats
# from sklearn import linear_model
# import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

#load data 
wd = os.getcwd()
data_directory = str(wd) + "\\Coding-Assess-main\\data\\"
input_file = "Part 2. loan_data_final.csv"
df = pd.read_csv(data_directory + input_file)
df1 = pd.read_csv(data_directory + input_file)
# Drop unnecessary columns
df.drop(columns=['Unnamed: 0'], inplace=True)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with median
num_imputer = SimpleImputer(strategy="median")
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute missing values in categorical columns with most frequent value (mode)
cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering: Create new features
df['debt_to_income_ratio'] = df['loan_percent_income'] * df['person_income']
df['credit_risk_factor'] = df['credit_score'] / df['cb_person_cred_hist_length']
df['loan_risk_score'] = df['borrower_risk_score'] * df['loan_int_rate']

#additonal features
# 1. Employment Stability Score
df['employment_stability'] = df['person_emp_exp'] / df['person_age']

# 2. Loan Burden Ratio
df['loan_burden_ratio'] = df['loan_to_income_ratio'] * df['person_income']

# 3. Credit Utilization Score
df['credit_utilization_score'] = df['credit_score'] / (df['loan_to_income_ratio'] + 1e-6)  # Avoid division by zero

# 4. Household Financial Burden (Avoiding division by zero)
df['household_financial_burden'] = df['loan_to_income_ratio'] / (df['dependents_count'] + 1)

# 5. Interest Rate Risk
df['interest_rate_risk'] = df['loan_int_rate'] * df['regional_unemployment_rate']

# 6. Credit Age Impact
df['credit_age_impact'] = df['cb_person_cred_hist_length'] / df['person_age']

# 7. Adjusted Risk Score
df['adjusted_risk_score'] = df['borrower_risk_score'] / (df['regional_unemployment_rate'] + 1e-6)  # Avoid zero division


# Drop original columns if they are now redundant
df.drop(columns=['loan_percent_income', 'credit_score', 'borrower_risk_score'], inplace=True)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display transformed data
print(df.head())