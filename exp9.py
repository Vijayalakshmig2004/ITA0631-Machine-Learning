import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('linear_regression_dataset.csv')
X = data[['Experience']]
y = data['Salary']

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, lin_model.predict(X), label='Linear', color='green')
plt.plot(X, poly_model.predict(X_poly), label='Polynomial', color='red')
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
