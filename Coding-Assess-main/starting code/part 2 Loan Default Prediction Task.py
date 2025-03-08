# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 18:22:10 2025

@author: morris
"""
import pandas as pd 
import numpy as np 
from fredapi import Fred
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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
# scaler = StandardScaler()
# numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])



# Train a Random Forest model to assess feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(df.drop(columns=['loan_status']), df['loan_status'])

# Get feature importances
feature_importances = rf_model.feature_importances_
feature_names = df.drop(columns=['loan_status']).columns

# Sort feature importances in descending order
sorted_idx = np.argsort(feature_importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(np.array(feature_names)[sorted_idx], feature_importances[sorted_idx], color='blue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance using Random Forest")
plt.gca().invert_yaxis()
plt.show()


print("Top 10 features using Random Forest:")
for i in range(10): 
    print(np.array(feature_names)[sorted_idx][i],)


# Compute permutation importance
perm_importance = permutation_importance(rf_model, df.drop(columns=['loan_status']), df['loan_status'], scoring='accuracy', n_repeats=10, random_state=42)

# Sort feature importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

# Plot permutation importance
plt.figure(figsize=(12, 6))
plt.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx], color='green')
plt.xlabel("Permutation Importance Score")
plt.ylabel("Features")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.show()
