import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
datasets = pd.read_csv(
    '/GitHub/MachineLearning/Simple_Linear_Regression/Salary.csv')

# Devide the datasets x and y
x = datasets.iloc[:, :-1]
y = datasets.iloc[:, 1]

# train & test data set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0)

# LInearRegression
from sklearn.linear_model import LinearRegression
simlpeLinearRegression = LinearRegression()
simlpeLinearRegression.fit(x_train, y_train)

# Prediction
y_predict = simlpeLinearRegression.predict(x_test)


# implement the graph
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, simlpeLinearRegression.predict(x_train))
plt.show()
