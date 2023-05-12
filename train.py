
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training data
training1_data = pd.read_csv('training1.csv')
training2_data = pd.read_csv('training2.csv')

# Concatenate the training data
training_data = pd.concat([training1_data, training2_data])

# Load the test data
test_data = pd.read_csv('test.csv')

# Extract features and labels from the training data
X_train = training_data.iloc[:, :-2]  # Exclude class labels and confidence
y_train = training_data.iloc[:, -2]   # Class labels
confidence_train = training_data.iloc[:, -1]  # Confidence labels

# Impute missing values in the training data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_imputed, y_train, test_size=0.2, random_state=42)

# Train a random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train_split, y_train_split)

# Predict the validation set labels
y_val_pred = classifier.predict(X_val_split)

# Calculate the accuracy on the validation set
accuracy = accuracy_score(y_val_split, y_val_pred)
print("Validation Accuracy:", accuracy)

# Impute missing values in the test data
X_test = test_data.iloc[:, :-1]  # Exclude confidence labels
X_test_imputed = imputer.transform(X_test)

# Predict the test set labels
y_test_pred = classifier.predict(X_test_imputed)

# Save the predictions to a file
np.savetxt('predictions.csv', y_test_pred, delimiter=',')

