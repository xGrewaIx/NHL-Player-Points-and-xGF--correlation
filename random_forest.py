import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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

# Initialize 200 decision trees, play around with this value
# More decision trees the bteer performance, as expected
forest = RandomForestRegressor(n_estimators=200)
forest.fit(X, y)
data['rf predicted total points'] = forest.predict(X)

# Plotting the Random Forest regression
plt.figure(figsize=(10, 6))
plt.scatter(y, data['rf predicted total points'], color='green', alpha=0.5, label='RF Predicted Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual Total Points')
plt.ylabel('Predicted Total Points')
plt.title('Regression Plot: Actual vs Predicted Total Points (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()

# R-squared for Random Forest
rf_rsquared_1 = r2_score(y, data['rf predicted total points'])
print(f"Random Forest R-squared Total Points: {rf_rsquared_1}")


# Step 1: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit the RandomForestRegressor on the training data
forest = RandomForestRegressor(n_estimators=200)
forest.fit(X_train, y_train)

# Step 3: Predict the values for the test set
y_pred = forest.predict(X_test)

# Step 4: Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared on test set for Total Points: {r2}')



print("#----------------------------------------------------------------------------------#")


print("xGF% Model")

corr_xGF_col = ['SCF%', 'HDCF%', 'FF%', 'SF%', 'CF%', 'MDCF%', 'LDCF%']

X = data[corr_xGF_col]
y = data['xGF%']

forest = RandomForestRegressor(n_estimators=200)
forest.fit(X, y)
data['rf predicted xGF%'] = forest.predict(X)

# Plotting the Random Forest regression
plt.figure(figsize=(10, 6))
plt.scatter(y, data['rf predicted xGF%'], color='green', alpha=0.5, label='RF Predicted xGF%')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, linestyle='--', label='Perfect Fit Line')
plt.xlabel('Actual xGF%')
plt.ylabel('Predicted xGF%')
plt.title('Regression Plot: Actual vs Predicted xGF% (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()

# R-squared for Random Forest
rf_rsquared_2 = r2_score(y, data['rf predicted xGF%'])
print(f"Random Forest R-squared xGF%: {rf_rsquared_2}")


# Step 1: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit the RandomForestRegressor on the training data
forest = RandomForestRegressor(n_estimators=200)
forest.fit(X_train, y_train)

# Step 3: Predict the values for the test set
y_pred = forest.predict(X_test)

# Step 4: Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared on test set for xGF%: {r2}')




"""
What is interesting is that Random Forest will always create 
a model that has a high R^2 value, but when tested for accuracy
it does not perform as well at times. 

Total points model has a R^2 value of 0.9655
but when split into training and testing data, the accuracy
score is 0.7731


I find that according to this article:
(https://www.keboola.com/blog/random-forest-regression#:~:text=The%20main%20problem%20is%20that,generalize%20well%20to%20novel%20data.)
Random forest has high variance and overfit the data, which leads to the low accuracy
in the testing stage above. 

Next step will be to try other models, and study more to create more indepth models.
"""