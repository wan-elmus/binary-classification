
import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load training data
training1_df = pd.read_csv("training1.csv")
training2_df = pd.read_csv("training2.csv")

# Load test data
test_df = pd.read_csv("test.csv")

# Handle missing values
training2_df.fillna(training2_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Feature selection
X_train1 = training1_df.iloc[:, 1:].values
y_train1 = training1_df.iloc[:, 0].values

X_train2 = training2_df.iloc[:, 1:].values
y_train2 = training2_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values

# Convert categorical variable into numerical 
y_train2[y_train2==1.0] = 1
y_train2[y_train2==0.0] = -1

# Convert labels to binary values
y_train1 = np.where(y_train1 >= 0.5, 1, 0)
y_train2 = np.where(y_train2 >= 0.5, 1, 0)

# Train logistic regression classifier on first dataset
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train1, y_train1)
    
# Handle missing values in second dataset
X_train2[np.isnan(X_train2)] = np.nanmean(X_train2)
y_train2[np.isnan(y_train2)] = np.nanmean(y_train2)

# Evaluate performance on second dataset
score = lr.score(X_train2, y_train2)
print("Accuracy on second dataset:", score)

### Find missing columns in both training datasets
missing_cols = set(training1_df.columns) | set(training2_df.columns) - set(test_df.columns)

# Add missing columns to test dataset and set their values to 0
for col in missing_cols:
    test_df[col] = 0


# Make predictions on test data using logistic regression
test_preds = lr.predict(X_test)

# Save test predictions to a file
with open('test_predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prediction'])
    for pred in test_preds:
        writer.writerow([pred])

# Define classifiers and their hyperparameters
svc = SVC()
svc_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}

# Perform grid search and cross-validation for each classifier
classifiers = {'SVM': (svc, svc_params), 'Random Forest': (rf, rf_params)}
results = {}
for name, (classifier, params) in classifiers.items():
    print("Performing grid search and cross-validation for", name)
    clf = GridSearchCV(classifier, params, cv=5)
    clf.fit(X_train1, y_train1)
    results[name] = clf

# Save results to a file
with open('results.pickle', 'wb') as f:
    pickle.dump(results, f)

# Scale the features
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_train2 = scaler.transform(X_train2)
X_test = scaler.transform(X_test)

# Select the top 10 features based on ANOVA F-value
selector = SelectKBest(f_classif, k=10)
X_train1 = selector.fit_transform(X_train1, y_train1)
X_train2 = selector.transform(X_train2)
X_test = selector.transform(X_test)

