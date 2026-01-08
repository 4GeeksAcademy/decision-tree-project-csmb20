# Import libraries

import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sn
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import chi2_contingency
import warnings
from IPython.display import display, Math
from pickle import dump

warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.3f}'.format
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# Import data
dfDiabetesClass_raw = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=421&path=diabetes.csv")

# Data inspection

print(dfDiabetesClass_raw.shape)
print(dfDiabetesClass_raw.info())

# Verify duplicates and NANs

if dfDiabetesClass_raw.duplicated().any().any():
    dfDiabetesClass_raw = dfDiabetesClass_raw.drop_duplicates()

if dfDiabetesClass_raw.isna().any().any():
    dfDiabetesClass_raw = dfDiabetesClass_raw.dropna()

# Exploratory data analysis

# Get outliers
fig, axis = plt.subplots(3,3,figsize=(10,7))
r = 0
c = 0
for col_name in dfDiabetesClass_raw.columns:
    sn.boxplot(ax=axis[r,c], data = dfDiabetesClass_raw, x = dfDiabetesClass_raw[col_name])
    c += 1
    if c > 2:
        r += 1
        c = 0

plt.suptitle('Boxplot pre-processed data')
plt.tight_layout()
plt.show()

# Remove some of the outliers

dfDiabetesClass_processed = dfDiabetesClass_raw
dfDiabetesClass_processed = dfDiabetesClass_processed[dfDiabetesClass_processed['Insulin'] <= 300]
dfDiabetesClass_processed = dfDiabetesClass_processed[dfDiabetesClass_processed['DiabetesPedigreeFunction'] <= 1.2]
dfDiabetesClass_processed = dfDiabetesClass_processed[dfDiabetesClass_processed['Age'] <= 65]
dfDiabetesClass_processed = dfDiabetesClass_processed[dfDiabetesClass_processed['BMI'] <= 50]
dfDiabetesClass_processed = dfDiabetesClass_processed[dfDiabetesClass_processed['BMI'] >= 18]

fig, axis = plt.subplots(3,3,figsize=(10,7))
r = 0
c = 0
for col_name in dfDiabetesClass_raw.columns:
    sn.boxplot(ax=axis[r,c], data = dfDiabetesClass_processed, x = dfDiabetesClass_processed[col_name])
    c += 1
    if c > 2:
        r += 1
        c = 0

plt.suptitle('Boxplot (outliers removed)')
plt.tight_layout() 
plt.show()

# Get correlation matrix

fig, ax = plt.subplots(figsize=(10,7))
sn.heatmap(dfDiabetesClass_processed.corr(method="pearson").abs(), annot=True, annot_kws={"fontsize": 10}, fmt=".2f", cmap="viridis", ax=ax)
plt.tight_layout()
plt.show()

# Get features based on correlation matrix (threshold > 0.2)
correlationMatrix     = dfDiabetesClass_processed.corr(method='pearson',numeric_only=True)
correlationMatrix_abs = correlationMatrix['Outcome'].abs()
feature_names         = correlationMatrix_abs.index[correlationMatrix_abs > 0.2].to_list()

print(f'The independent variables that explain the variable "Outcome" are: {feature_names[:-1]}')

# Verify that features do not have high correlation

dfDiabetesClass_processed_featured = dfDiabetesClass_processed[feature_names]
dfDiabetesClass_processed_featured = dfDiabetesClass_processed_featured.drop(columns=['Outcome'])
feature_correlation = dfDiabetesClass_processed_featured.corr(method="pearson")
print(feature_correlation)

fig, ax = plt.subplots(figsize=(10,7))
sn.heatmap(dfDiabetesClass_processed_featured.corr(method="pearson").abs(), annot=True, fmt=".2f", cmap="viridis", ax=ax)
plt.tight_layout()
plt.show()

print("Age and pregnancies have a correlation of 0.531, which is larger than the threshold of 0.5.")

# # Exclusion of independent variables based on the feature correlation matrix
features      = feature_names[1:]  # Exclude 'Pregnancies' due to high correlation with 'Age'
features      = feature_names[:-1]  # Exclude target variable

target   = 'Outcome'

# Set up the data

X      = dfDiabetesClass_processed[features]
y      = dfDiabetesClass_processed[target]

# To test best split trade-off we loop over different splitting ratios

accuracy  = []
precision = []
recall    = []
f1        = []


for split_ratio in np.arange(0.5,1,0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = split_ratio)

    # Decision Tree Classifier

    diabetes_model = DecisionTreeClassifier() 
    diabetes_model.fit(X_train, y_train)

    # Perform prediction
    y_pred = diabetes_model.predict(X_test)

    # Compute accuracy, precision, and recall
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

# Plot split vs. scores

fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,figsize=(15,7))

ax0.plot(np.arange(0.5,1,0.1),accuracy,color='blue',linewidth=2)
ax0.grid()
ax0.set_xlabel("Data split (train percentage) [%]")
ax0.set_ylabel("Accuracy [%]")
ax0.set_title("Model accuracy")
ax1.plot(np.arange(0.5,1,0.1),precision,color='red',linewidth=2)
ax1.grid()
ax1.set_xlabel("Data split (train percentage) [%]")
ax1.set_ylabel("Precision [%]")
ax1.set_title("Model precision")
ax2.plot(np.arange(0.5,1,0.1),recall,color='green',linewidth=2)
ax2.grid()
ax2.set_xlabel("Data split (train percentage) [%]")
ax2.set_ylabel("Recall [%]")
ax2.set_title("Model recall")
ax3.plot(np.arange(0.5,1,0.1),f1,color='purple',linewidth=2)
ax3.grid()
ax3.set_xlabel("Data split (train percentage) [%]")
ax3.set_ylabel("F1 Score [%]")
ax3.set_title("Model F1 Score")

plt.tight_layout()
plt.show()

# To test best tree max depth, we loop over different tree depths

best_split = 0.7  # From previous analysis

accuracy_train  = []
accuracy_test   = []
precision_train = []
precision_test  = []
recall_train    = []
recall_test     = []
f1_train        = []
f1_test         = []

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = best_split)

tree_depth_line = np.linspace(1,10).astype(int)

for tree_depth in tree_depth_line:
    # Decision tree classifier

    diabetes_model = DecisionTreeClassifier(max_depth=tree_depth)
    diabetes_model.fit(X_train, y_train)

    # Perform prediction
    y_pred_train = diabetes_model.predict(X_train)
    y_pred_test = diabetes_model.predict(X_test)

    # Compute accuracy, precision, and recall
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    accuracy_test.append(accuracy_score(y_test, y_pred_test))
    precision_train.append(precision_score(y_train, y_pred_train))
    precision_test.append(precision_score(y_test, y_pred_test))
    recall_train.append(recall_score(y_train, y_pred_train))
    recall_test.append(recall_score(y_test, y_pred_test))
    f1_train.append(f1_score(y_train, y_pred_train))
    f1_test.append(f1_score(y_test, y_pred_test))

# Plot split vs. scores

fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,figsize=(15,7))

ax0.plot(tree_depth_line,accuracy_train,color='blue',linewidth=2)
ax0.plot(tree_depth_line,accuracy_test,color='orange',linewidth=2)
ax0.grid()
ax0.set_xlabel("Tree depth")
ax0.set_ylabel("Accuracy [%]")
ax0.set_title("Model accuracy")
ax1.plot(tree_depth_line,precision_train,color='blue',linewidth=2)
ax1.plot(tree_depth_line,precision_test,color='orange',linewidth=2)
ax1.grid()
ax1.set_xlabel("Tree depth")
ax1.set_ylabel("Precision [%]")
ax1.set_title("Model precision")
ax2.plot(tree_depth_line,recall_train,color='blue',linewidth=2)
ax2.plot(tree_depth_line,recall_test,color='orange',linewidth=2)
ax2.grid()
ax2.set_xlabel("Tree depth")
ax2.set_ylabel("Recall [%]")
ax2.set_title("Model recall")
ax3.plot(tree_depth_line,f1_train,color='blue',linewidth=2)
ax3.plot(tree_depth_line,f1_test,color='orange',linewidth=2)
ax3.grid()
ax3.set_xlabel("Tree depth")
ax3.set_ylabel("F1 Score [%]")
ax3.set_title("Model F1 Score")

plt.tight_layout()
plt.show()

# Choosing best split ratio = 50%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = best_split)

# Tree classification

best_tree_depth = 3

diabetes_model = DecisionTreeClassifier(max_depth=best_tree_depth)
diabetes_model.fit(X_train, y_train)

# Perform prediction
y_pred = diabetes_model.predict(X_test)

# Compute accuracy, precision, recall, and f1 score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'The accuracy score of the model is: {accuracy}\n')
print(f'The precision score of the model is: {precision}\n')
print(f'The recall score of the model is: {recall}\n')
print(f'The F1 score of the model is: {f1}')

# Hyperparameter optimization

hyperparams = {
    "max_depth": [None, 1, 2, 3, 4, 5],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 5, 10, 15]
}

# We initialize the grid
grid = GridSearchCV(diabetes_model, hyperparams, scoring = 'f1', cv = 5)
grid.fit(X_train, y_train)

# Get best parameters
print(f"Best hyperparameters: {grid.best_params_}")

# Test best hyperparameter combination

diabetes_model = DecisionTreeClassifier(**grid.best_params_)
diabetes_model.fit(X_train, y_train)
y_pred = diabetes_model.predict(X_test)

# Compute accuracy, precision, and recall
grid_accuracy   = accuracy_score(y_test, y_pred)
grid_precision  = precision_score(y_test, y_pred)
grid_recall     = recall_score(y_test, y_pred)
grid_f1         = f1_score(y_test, y_pred)

print('The best linear regression model to predict the health factor uses the following independent variables\n')
for i, col in enumerate(features):
    print(f'{i+1}. {col}')
print('\n')

print(f'The linear regression after processing the data has the following performance parameters:\n')
print(f'Accuracy = {grid_accuracy:0.3f}')
print(f'Precision = {grid_precision:0.3f}')
print(f'Recall = {grid_recall:0.3f}')
print(f'F1 Score = {grid_f1:0.3f}\n')

"""
The model prediction capability is rather limited due to the fact that features used to predict the diabetes outcome do not have a strong correlation 
with the target variable. The fact that the precision of the model is lower than the recall indicates that the model is better at identifying true positives
(diabetic patients) than avoiding false positives (non-diabetic patients classified as diabetic). 
This is confirmed by the F1 score being closer to the recall than to the precision.

More relevant features are required to improve the model performance.
"""

fig = plt.figure(figsize=(15,10))
plot_tree(diabetes_model, feature_names=features, class_names=['No Diabetes','Diabetes'], filled=True)
plt.show()

dump(diabetes_model, open("decision_tree_classifier_default_42.sav", "wb"))
