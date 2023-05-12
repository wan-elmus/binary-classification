import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load training data
training1_df = pd.read_csv("training1.csv")
training2_df = pd.read_csv("training2.csv")

# Concatenate the training datasets
training_df = pd.concat([training1_df, training2_df])

# Load test data
test_df = pd.read_csv("test.csv")

# Handle missing values
training_df.fillna(training_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Feature selection
X = training_df.iloc[:, 1:].values
y = training_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values

# Convert categorical variable into numerical 
y[y==1.0] = 1
y[y==0.0] = -1

# Convert labels to binary values
y = np.where(y >= 0.5, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression classifier
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Evaluate performance on test dataset
score = lr.score(X_test, y_test)
print("Accuracy on test dataset:", score)

## Find missing columns in both training datasets
missing_cols = set(training1_df.columns) | set(training2_df.columns) - set(test_df.columns)

# Add missing columns to test dataset and set their values to 0
missing_cols_df = pd.DataFrame(0, index=np.arange(test_df.shape[0]), columns=list(missing_cols))

test_df = pd.concat([test_df, missing_cols_df], axis=1)

# Make predictions on test data using logistic regression
test_preds = lr.predict(X_test)

# Save test predictions to a file
with open('predictions.csv', 'w', newline='') as file:
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
    clf.fit(X_train, y_train)
    results[name] = clf

# Save results to a file
with open('results.pickle', 'wb') as f:
    pickle.dump(results, f)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Select the top 10 features based on ANOVA F-value
selector = SelectKBest(f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

with open('results.pickle', 'rb') as f:
    results = pickle.load(f)

# Print the best hyperparameters and corresponding score for SVM
print("Best parameters for SVM:", results['SVM'].best_params_)
print("Best score for SVM:", results['SVM'].best_score_)

# Print the best hyperparameters and corresponding score for Random Forest
print("Best parameters for Random Forest:", results['Random Forest'].best_params_)
print("Best score for Random Forest:", results['Random Forest'].best_score_)