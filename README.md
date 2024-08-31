# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Load the data
3. Perform iterations og gradient steps with learning rate.
4. Print the predictions with state names

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Gnanendran N
RegisterNumber: 212223240037
```
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
file_path = '50_Startups.csv'
data = pd.read_csv(file_path)
print("Original Data:")
print(data.head())
encoder = OneHotEncoder(drop='first')  # drop='first' to avoid multicollinearity
state_encoded = encoder.fit_transform(data[['State']]).toarray()
state_encoded_df = pd.DataFrame(state_encoded, columns=encoder.get_feature_names_out(['State']))
data_processed = pd.concat([data.drop('State', axis=1), state_encoded_df], axis=1)
X = data_processed.drop('Profit', axis=1).values
y = data_processed['Profit'].values
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]
theta = np.zeros(X_normalized.shape[1])
learning_rate = 0.01
num_iterations = 1500

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):
        predictions = X.dot(theta)
        theta = theta - (1/m) * learning_rate * (X.T.dot(predictions - y))
        cost_history[i] = compute_cost(X, y, theta)        
    return theta, cost_history

theta, cost_history = gradient_descent(X_normalized, y, theta, learning_rate, num_iterations)
print("\nLearned Parameters (theta):")
print(theta)
predicted_profits = X_normalized[:3].dot(theta)
states = data['State'][:3]
print("\nPredicted Profits for the First 3 States:")
for state, profit in zip(states, predicted_profits):
    print(f"State: {state}, Predicted Profit: ${profit:.2f}")
```
## Output:
![image](https://github.com/user-attachments/assets/6dfc5121-044f-45f2-9bf2-9ad170d58de1)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
