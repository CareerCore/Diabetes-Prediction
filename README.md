# Diabetes Prediction using Machine Learning

This project is a Machine Learning-based Diabetes Prediction System developed using Python and Support Vector Machine (SVM). The model predicts whether a person is diabetic or not based on medical attributes.

---

## 📌 Project Overview

The project uses the Diabetes Dataset and applies data preprocessing, feature scaling, train-test splitting, and SVM classification to predict diabetes.

The system takes input features such as:

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

and predicts whether the person is diabetic or non-diabetic.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

---

## 📚 Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score# Diabetes-Prediction
