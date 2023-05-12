
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load training data
training1_df = pd.read_csv("training1.csv")
training2_df = pd.read_csv("training2.csv")

# Load test data
test_df = pd.read_csv("test.csv")

# Define function for preprocessing data
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Feature selection
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Convert categorical variable into numerical variable
    y[y==1.0] = 1
    y[y==0.0] = -1
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Preprocess training data
X_train1_pre, y_train1_pre = preprocess_data(training1_df)
X_train2_pre, y_train2_pre = preprocess_data(training2_df)
X_test_pre, _ = preprocess_data(test_df)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define classifiers and their hyperparameters
lr = LogisticRegression(random_state=42)
svc = SVC()
svc_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}

# Perform grid search and cross-validation for each classifier
classifiers = {'Logistic Regression': (lr, {}), 'SVM': (svc, svc_params), 'Random Forest': (rf, rf_params)}
for name, (classifier, params) in classifiers.items():
    print("Performing grid search and cross-validation for", name)
    clf = GridSearchCV(classifier, params, cv=5)
    clf.fit(X_train1_pre, y_train1_pre)
    print("Best parameters:", clf.best_params_)
    print("Training score:", clf.best_score_)
    print("Validation score:", clf.score(X_train2_pre, y_train2_pre))
    
    # Evaluate on test data
    y_pred = clf.predict(X_test_pre)
    
    # further analysis and evaluation of the logistic regression model output here
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate logistic regression model on test data
lr_clf = classifiers['Logistic Regression'][0]
lr_clf.set_params(**clf.best_params_)
lr_clf.fit(X_train1_pre, y_train1_pre)
y_pred = lr_clf.predict(X_test_pre)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
