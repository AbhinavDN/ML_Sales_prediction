# Sales Prediction Using Machine Learning
![ML_logo](https://github.com/AbhinavDN/ML_Sales_prediction/blob/main/ML(Overview).png)

## End-To-End Workflow 
![](https://github.com/AbhinavDN/ML_Sales_prediction/blob/main/End-to-end%20ML%20project%20workflow(Img).jpg)

## 1️⃣ Project Overview

This project focuses on predicting product sales using a Machine Learning Linear Regression model. The dataset contains advertising budgets spent on different platforms such as TV, Radio, and Newspaper, and the corresponding Sales generated.

The objective is to analyze how advertising investments impact sales and to build a predictive model that can estimate future sales based on advertising budgets.

The project includes:

Data loading and preprocessing

Exploratory Data Analysis (EDA)

Model building using Linear Regression

Model evaluation and prediction

## 2️⃣ Objective of the Project

The main objectives of this project are:

To analyze the relationship between advertising budget and sales.

To build a Linear Regression model for predicting sales.

To evaluate model performance using appropriate metrics.

To understand which advertising medium contributes most to sales.

To provide a simple and effective sales forecasting solution.

## 3️⃣ Project Schema (Working Process)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load Dataset
df = pd.read_csv("/content/Advertising Budget and Sales.csv")
df.head()

## Drop unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

df

## Define Features (X) and Target (Y)
X = df.drop(columns=["Sales"])
X.head()

Y = df["Sales"]
Y.head()

## Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, ytest = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

x_train.shape

## Linear Regression Model
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

## Train Model
linear_reg.fit(x_train, y_train)

## Model Parameters
linear_reg.intercept_
linear_reg.coef_

## Regression Equation
 m1 = 0.04458402
 m2 = 0.19649703
 m3 = -0.00278146
 Sales = (m1 * TV) + (m2 * Radio) + (m3 * Newspaper) + c

## Prediction on Test Data
y_pred = linear_reg.predict(x_test)

## Model Evaluation
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)
mse

## Root Mean Squared Error (RMSE)
np.sqrt(mse)

## Sample Prediction
linear_reg.predict([[230.1, 37.8, 69.2]])

## Install Gradio (for UI)
%%capture
pip! install gradio

## Gradio Interface
import gradio as gr

def sales_predictor(TV, Radio, Newspaper):
    prediction = linear_reg.predict([[TV, Radio, Newspaper]])
    return float(prediction[0])

demo = gr.Interface(
    fn=sales_predictor,
    inputs=["number", "number", "number"],
    outputs="number"
)

demo.launch()

## Ignore warnings
import warnings
warnings.filterwarnings("ignore")

###🔹 Step 1: Data Collection

Dataset: Advertising Budget and Sales.csv

Features:

TV Advertising Budget

Radio Advertising Budget

Newspaper Advertising Budget

Target Variable:

Sales

###🔹 Step 2: Data Preprocessing

Importing required libraries (NumPy, Pandas, Matplotlib, Scikit-learn)

Checking for:

Missing values

Data types

Basic statistics

Splitting data into:

Independent variables (X)

Dependent variable (Y)

###🔹 Step 3: Exploratory Data Analysis (EDA)

Checking correlation between advertising platforms and sales

Visualizing relationships using scatter plots

Understanding trends in data

###🔹 Step 4: Model Building

Splitting data into:

Training set

Testing set

Applying Linear Regression algorithm

Training the model using training data

###🔹 Step 5: Model Evaluation

Predicting sales on test data

Calculating:

R² Score

Mean Squared Error (MSE)

Comparing actual vs predicted sales

##4️⃣ Findings

Based on model training and analysis:

TV advertising shows a strong positive correlation with sales.

Radio advertising also has a significant impact on sales.

Newspaper advertising has comparatively less impact.

The Linear Regression model provides good prediction accuracy.

The R² score indicates that a large percentage of sales variation is explained by advertising budgets.

Key Insight:
👉 Increasing budget on TV and Radio advertisements can significantly improve sales performance.

##5️⃣ Conclusion

This project successfully demonstrates how Machine Learning can be used for sales forecasting.

Linear Regression effectively models the relationship between advertising spend and sales.

The model helps businesses make data-driven decisions.

Advertising investments, especially in TV and Radio, directly influence sales growth.

The solution can be extended using advanced algorithms for better accuracy.

Overall, the project proves that predictive analytics is a powerful tool for business growth and strategic planning.


