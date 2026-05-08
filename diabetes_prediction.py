# Diabetes Prediction using Machine Learning

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Display first 5 rows
print(diabetes_dataset.head())

# Dataset information
print("\nShape of Dataset:", diabetes_dataset.shape)

# Statistical measures
print("\nDataset Description:")
print(diabetes_dataset.describe())

# Checking output values
print("\nOutcome Counts:")
print(diabetes_dataset['Outcome'].value_counts())

# Mean values grouped by Outcome
print("\nGrouped Mean Values:")
print(diabetes_dataset.groupby('Outcome').mean())

# Separating features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print("\nFeatures:")
print(X)

print("\nTarget:")
print(Y)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

print("\nStandardized Data:")
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

print("\nDataset Shapes:")
print(X.shape, X_train.shape, X_test.shape)

# Creating the SVM Classifier
classifier = svm.SVC(kernel='linear')

# Training the model
classifier.fit(X_train, Y_train)

# Accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("\nAccuracy score of training data:", training_data_accuracy)

# Accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of test data:", test_data_accuracy)

# Predictive System
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

print("\nStandardized Input Data:")
print(std_data)

# Prediction
prediction = classifier.predict(std_data)

print("\nPrediction Result:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
