#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:14:57 2023

@author: christian
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as skl

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve

import xgboost as xgb 

from catboost import CatBoostClassifier, Pool, cv

from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

bank_train = pd.read_csv('/Users/christian/Downloads/archive (5)/train.csv')

bank_train.head()

bank_train.info()

bank_train.describe().transpose()

bank_train.isna().sum()

bank_train.duplicated().sum()

bank_train.shape

bank_train.select_dtypes(include=('int64')).nunique()

bank_train.select_dtypes(include=('object')).nunique()

numeric_columns = bank_train.select_dtypes(include=('int64'))
numeric_columns.hist(bins=20, figsize=(15,10))
plt.show()

sns.pairplot(bank_train[['age', 'balance', 'day', 'duration', 'campaign']], diag_kind='kde')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='duration', data=bank_train)
plt.title("Comparison of 'duration' for Subscribed ('yes') & Unsubscribed ('no)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='balance', data=bank_train)
plt.title("Comparison of 'balance' for Subscribed ('yes') & Unsubscribed ('no')")
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Countplots of Categorical Features")

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for i, feature in enumerate(categorical_features):
    row, col = i // 3, i % 3 
    sns.countplot(x=feature, hue='y', data=bank_train, ax=axes[row, col])
    
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=bank_train)
plt.title('Distribution of the Target Variable (y)')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=bank_train, x='education', palette='viridis')
plt.title('Count of General Health Status')
plt.xlabel('General Health Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

crosstab = pd.crosstab(bank_train['education'], bank_train['poutcome'])
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Reds')
plt.xlabel('P_outcome')
plt.ylabel('Education')
plt.show()

crosstab = pd.crosstab(bank_train['y'], bank_train['job'])
crosstab.plot(kind='area', colormap='viridis', alpha=0.7, stacked=True)
plt.title('Stacked Area Chart: Job Category by BankTermDeposit')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()

bank_encoded = bank_train.copy()

label_encoder = LabelEncoder()

for column in bank_encoded.select_dtypes(include='object'):
    bank_encoded[column] = label_encoder.fit_transform(bank_encoded[column])
    
bank_encoded.head()

bank_encoded.select_dtypes(include='int64').nunique()

correlation_matrix = bank_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Portugal Bank Deposit Correlation Heatmap")
plt.show()

print(correlation_matrix)

correlation_threshold = 0.05

low_corr_features = bank_encoded.columns[abs(bank_encoded.corr()['y']) < correlation_threshold]

print("Features to be removed:", low_corr_features)

bank_train_filtered = bank_encoded.drop(low_corr_features, axis=1)

print("\nDataFrame after feature removal:")
print(bank_train_filtered.head())

selected_columns = ['education', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_selected = bank_encoded[selected_columns]

X = bank_selected.drop(columns=['y']) #predictors
Y = bank_selected['y'] #response

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape", Y_train.shape)
print("Y_test shape", Y_test.shape)

class_distribution = Y_train.value_counts()
print("Class Distribution:")
print(class_distribution)

imbalance_ratio = class_distribution[0] / class_distribution[1]
print("Imbalance Ratio:", imbalance_ratio)

threshold = 2
if imbalance_ratio > threshold:
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    
    resampled_class_distribution = Y_train_resampled.value_counts()
    print("\nClass Distribution After SMOTE:")
    print(resampled_class_distribution)
    
    X_train = X_train_resampled
    Y_train = Y_train_resampled
    print("\nSMOTE Applied. Training Data Resampled")
else:
    print("\nNo Significant Class Imbalance. SMOTE Not Applied.")
  
selected_columns = ['education', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

Q1 = X_train[selected_columns].quantile(0.25)
Q3 = X_train[selected_columns].quantile(0.75)
IQR = Q3 - Q1

threshold = 1.5

outlier_mask = (
    (X_train[selected_columns] < (Q1 - threshold * IQR)) |
    (X_train[selected_columns] > (Q3 + threshold * IQR))
    ).any(axis=1)

X_train_clean = X_train[~outlier_mask]
Y_train_clean = Y_train[~outlier_mask]

num_rows_removed = len(X_train) - len(X_train_clean)
print(f"Number of Rows Removed Due to Outliers: {num_rows_removed}")

#test options and evaluation metric
num_folds = 10
seed = 42
scoring = 'neg_mean_squared_error'

#spot check algorithms
models = []
models.append(('LIR', LinearRegression()))
models.append(('LOR', LogisticRegression()))
models.append(('LASSO', Lasso()))
models.append(('RIDGE', Ridge()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('XGBOOST', XGBClassifier()))
models.append(('CATBOOST', CatBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('LGBM', LGBMClassifier()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=(seed), shuffle=(True))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#compare algorithms
fig = plt.figure()
fig.suptitle('Portugal Bank Deposit Modeling Algorithm Comparison')
ax = fig.add_subplot(111)
sns.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Kaggle user is right. XGBOOST Wins

xgb = XGBClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    }

grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_clean, Y_train_clean)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best Parameters:", best_params)

Y_pred = best_estimator.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

bank_test = pd.read_csv('/Users/christian/Downloads/archive (5)/test.csv')

bank_test.info()

bank_test_selected = bank_test[selected_columns]

bank_test_selected.info()

test_best_params = {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}

test_best_estimator = XGBClassifier(
    learning_rate=test_best_params['learning_rate'],
    max_depth=test_best_params['max_depth'],
    n_estimators=test_best_params['n_estimators'],
    subsample=test_best_params['subsample'],
    random_state=42
    )

test_best_estimator.fit(X_train_clean, Y_train_clean)

bank_test_encoded = bank_test.copy()

for column in bank_test_encoded.select_dtypes(include='object'):
    bank_test_encoded[column] = label_encoder.fit_transform(bank_test_encoded[column])
    
bank_test_selected = bank_test_encoded[selected_columns]

Y_test_pred = test_best_estimator.predict(bank_test_selected)

print("Predictions on the test dataset:")
print(Y_test_pred)

results_bank = pd.DataFrame({'Actual': bank_test['y'], 'Predicted': Y_test_pred})
print(results_bank)

results_bank['Actual'] = results_bank['Actual'].map({'no': 0, 'yes': 1})
results_bank['Predicted'] = results_bank['Predicted'].astype(int)

accuracy = accuracy_score(results_bank['Actual'], results_bank['Predicted'])

print("Model Accuracy:", accuracy) #Accuracy: 79.8% ~80%