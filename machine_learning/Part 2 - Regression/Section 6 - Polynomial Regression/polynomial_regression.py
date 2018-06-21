# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset (Note: make sure X is seen as a matrix)
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Plot the position number vs salary to see if the relationship is linear or not
# We see that the relationship is exponential and therefore non-linear
plt.scatter(X, y, color = 'red')

# Splitting the dataset into the Training set and Test set - no need as only 10 observations and need all data points
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling - no need as polynomial regression scales features in the package 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)
accuracy_linear = np.sqrt(metrics.mean_squared_error(y, lin_reg.predict(X)))
print("Accuracy of Linear Regression with linear features is: %(accuracy)d" % {"accuracy": accuracy_linear})

# Fitting Polynomial Regression to the dataset (it adds the intercept for you)
# to change the fits you add or remove degrees of polynomials
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# Now fit a linear regression to the new matrix X_poly
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
accuracy_poly = np.sqrt(metrics.mean_squared_error(y, lin_reg_2.predict(X_poly)))
print("Accuracy of Linear Regression with polynomial features is: %(accuracy)d" % {"accuracy": accuracy_poly})

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
y_linear_predicted = lin_reg.predict(X)
plt.plot(X, y_linear_predicted, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
y_poly_predicted = lin_reg_2.predict(poly_reg.fit_transform(X))
plt.plot(X, y_poly_predicted, color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# [OPTIONAL] Visualising the Polynomial Regression results (for higher resolution and 
# smoother curve - smaller incremental steps in poition level)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Now predict a new result with our final Polynomial Regression model
new_position_level = 6.5

# Convert this new level to the polynomial features
poly_position_level = poly_reg.fit_transform(new_position_level)

# Predict the new output based on the polynomial features
y_pred = lin_reg_2.predict(poly_position_level)