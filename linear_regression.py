import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""
I utilized/learned these machine learning models from the book
"Python Data Science Handbook" by Jake VaderPlas
"""


data = pd.read_csv("All_Player_Data.csv")

print("Total Points Model")
corr_TP_col = ['iSCF', 'Shots', 'iFF', 'ixG', 'Off. Zone Starts', 'iCF',
               'Takeaways', 'iHDCF']

X = data[corr_TP_col]
y = data['Total Points']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
data['predicted total points'] = model.predict(X)
y_model = model.predict(X)

# Plotting the regression
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted total points
plt.scatter(y, data['predicted total points'], color='blue', alpha=0.5, label='Predicted Points')

# Plot the regression line (since it's simple linear regression, we plot the diagonal for perfect prediction)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, linestyle='--', label='Perfect Fit Line')

# Adding labels and title
plt.xlabel('Actual Total Points')
plt.ylabel('Predicted Total Points')
plt.title('Regression Plot: Actual vs Predicted Total Points')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

dlr_rsquared_1 = r2_score(y, data['predicted total points'])
print(f"Linear Regression R-squared: {dlr_rsquared_1}")


# Now I will split data into Training and testing data sets and check that R^2 value
# Step 1: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit the Linear Regression on the training data
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Step 3: Predict the values for the test set
y_pred = model.predict(X_test)

# Step 4: Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared on test set for Total Points: {r2}')


print("#----------------------------------------------------------------------------------#")


print("xGF% Model")

corr_xGF_col = ['SCF%', 'HDCF%', 'FF%', 'SF%', 'CF%', 'MDCF%', 'LDCF%']

X = data[corr_xGF_col]
y = data['xGF%']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
data['predicted xGF%'] = model.predict(X)

# Plotting the regression
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted total points
plt.scatter(y, data['predicted xGF%'], color='blue', alpha=0.5, label='Predicted xGF%')

# Plot the regression line (since it's simple linear regression, we plot the diagonal for perfect prediction)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, linestyle='--', label='Perfect Fit Line')

# Adding labels and title
plt.xlabel('Actual xGF%')
plt.ylabel('Predicted xGF%')
plt.title('Regression Plot: Actual vs Predicted xGF%')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

dlr_rsquared_2 = r2_score(y, data['predicted xGF%'])
print(f"Linear Regression R-squared (xGF%): {dlr_rsquared_2}")

# Step 1: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit the Linear Regression on the training data
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

# Step 3: Predict the values for the test set
y_pred = model.predict(X_test)

# Step 4: Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared on test set for xGF%: {r2}')