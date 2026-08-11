# Employee Churn Prediction

Employee churn prediction using Machine Learning and Streamlit.

## 📌 Project Overview

This project predicts whether an employee is likely to leave an organization based on various employee-related factors.

A Machine Learning classification model is trained on employee data and deployed through an interactive Streamlit web application.

## 🚀 Features

- Predicts whether an employee is likely to leave
- Provides an estimated probability of employee attrition
- Interactive Streamlit interface
- Handles numerical and categorical employee information
- Uses feature scaling and preprocessing
- Compares Logistic Regression with Decision Tree classification
- Evaluates models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## 🧠 Machine Learning Workflow

1. Load the employee dataset using Pandas
2. Inspect and clean the data
3. Convert the target variable into numerical values
4. Encode categorical features using one-hot encoding
5. Separate features (`X`) and target (`y`)
6. Split the dataset into training and testing sets
7. Scale numerical features using `StandardScaler`
8. Train a Logistic Regression model
9. Compare its performance with a Decision Tree model
10. Evaluate the models using classification metrics
11. Save the trained model and scaler using Joblib
12. Use the saved model in the Streamlit application

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## 📊 Model

The project uses **Logistic Regression** as the finalized prediction model.

A **Decision Tree Classifier** is also trained and evaluated for comparison.

The Logistic Regression model is saved as:

`logistic_model.pkl`

The feature scaler is saved as:

`scaler.pkl`

The feature column information is saved as:

`feature_columns.pkl`

## 💻 Streamlit Application

The Streamlit application allows users to enter employee details such as:

- Age
- Daily Rate
- Distance From Home
- Education
- Job Involvement
- Job Level
- Job Satisfaction
- Monthly Income
- Monthly Rate
- Number of Companies Worked
- Overtime
- Performance Rating
- Relationship Satisfaction
- Stock Option Level
- Total Working Years
- Training Times Last Year
- Work-Life Balance
- Years At Company
- Years In Current Role
- Years Since Last Promotion
- Years With Current Manager
- Business Travel
- Department
- Education Field
- Gender
- Job Role
- Marital Status

The application processes the entered information and returns a prediction along with the estimated probability of employee attrition.

##

TECHNOLOGIES USED

Python

Pandas

NumPy

Scikit-learn

Joblib

Streamlit

Spyder

GitHub

##

PROJECT STRUCTURE

text

employee_churn_project/

app.py

projectl.py

logistic_model.pkl

scaler.pkl

feature_columns.pkl

README.nd

gitattributes
