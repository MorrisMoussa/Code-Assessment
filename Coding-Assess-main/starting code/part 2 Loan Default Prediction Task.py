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
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### Documentation of Approach
documentation = """ 
     Objectivce: Build a machine learning model to predict loan defaults using the provided datase
    
     Since this machine learning model is predicting a binary outcome: default vs not default, 
     a classifier and/or logitic regression model is approriate for this situation 
    
     1. Loan in the data, and account for any missing values.
    
     2. Feature Engineering: there are multiple addtional features that could be calculated 
     from the oringal dataset which might have significant predictive power. For example, their credit
    utilization will one would assume gives insight into potential financial strain (and hence default probabilty
    on their loan). Additonally, debt_to_income_ratio is another feature that can be calculated based
    on the orignal set of features that also gives an indication of a borrower's  capacity to pay their
    periodic debt payment given their periodic income.  
    
     Based on the orginal feature set, and the additonal features created, run a classifier algorithm (ranndom forest)
    and rank feature importance/predictive power. To limit overfitting and keeping the model rebust, we can
    keep the top 10 most important features to train our final model. 
    
    3. Model Training: Seperate our top 10 features (X) and our target variable (Y). The Features are
    standardized so they are on the same order of magnitude/scale. Split data into training data (80%) 
    set to train the model and test data (20%) set.
    
    4. Summary: 
    Using the model that was trained using the training set, the test data set was used to make predictions. 
    These predictions were then compared to the actual outcomes of the test set by creating a Confusion 
    Matrix that shows True Positives, True Negatives, False Positives, False Negatives. These can be 
    used to calculate Precision (Positive Predictive Value)
    , Recall (Sensitivity / True Positive Rate)
    , F1-Score (Harmonic Mean of Precision & Recall)
    and Accuracy (over all correctness of the mode). 
    
    Since we are trying to predict probabilty of loan default, having false
    negatives is more detramental. Therefore, having a higher Recall 
    is more valuable from a model. While Accuracy is a more overall measure
    and therefore also important. 
    
    Givent these facts, the Recall and Accuracy of the Random Forest was 
    higher, and hence is likeliy the better model to predect loan 
    default. 
    
"""
print(documentation)


######## Load data & handle missing values ##########################
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
    
    
######## Feature Engineering ##########################
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
#select top 10 features
selected_features = np.array(feature_names)[sorted_idx][0:10]

######## Model Training ##########################
# Define features (X) and target (y)
X = df.drop(columns=['loan_status'])
X = X.loc[:,selected_features]
y = df['loan_status']

# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Evaluate Random Forest
print("Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))