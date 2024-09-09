## Information about this repository

This repository utilizes Linear Regression and Random Forest to create models that predict a players Total Points and xGF% (Expected goals for percentage).

**Data was obtained from Natural Stat Trick (https://www.naturalstattrick.com)**

## Findings

Linear Regression: The R^2 values for both the model and test set are both similar. The total points model R^2 value was ~0.7717, and the xGF% model was ~0.9515. 
Random Forest: The R^2 values for both the model and test set were similar for the xGF% model but differed in the total points model. 
- What is interesting is that Random Forest will always create  a model that has a high R^2 value, but when tested for accuracy it does not perform as well at times. Total points model has a R^2 value of 0.9655 but when split into training and testing data, the accuracy score is 0.7731. I find that according to this article: (https://www.keboola.com/blog/random-forest-regression#:~:text=The%20main%20problem%20is%20that,generalize%20well%20to%20novel%20data.) Random forest has high variance and overfit the data, which leads to the low accuracy in the testing stage above. 

## Up to date Step by Step process of what I have completed

- I obtained all the data from Natural Stat Trick and combined it into 1 csv file **(All_Player_Data.csv)**.
- I created a correlation matrix to find what statistics were highly correlated with Total Points and xGF%.
  - I created visualizations for each one of these statistics. 
- Created predictive models utilizing Linear Regression and Random Forest.
- Analyzed those models and tested them by creating testing and training sets. 

## Future Steps I will take to enhance my analysis 

- Next step will be to try other models, and study more to create more indepth models.
