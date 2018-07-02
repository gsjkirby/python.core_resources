# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling - needed as SVR does not include feature scaling in the class
# Reshape y to be a 2D array and not 1D
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y = y.astype(float).reshape(-1, 1)
y_scaled = sc_y.fit_transform(y)

# Fitting SVR to the dataset, choose rbf kernel for non-linear problem
regressor = SVR(kernel = 'rbf')
regressor.fit(X_scaled, y_scaled)

# Predicting a new result
y_pred_scaled = regressor.predict(6.5)

# Make sure to inverse the scaling to be able to see actual predicted value
y_pred = sc_y.inverse_transform(y_pred_scaled)

# Visualising the SVR results (scaled)
plt.scatter(X_scaled, y_scaled, color = 'red')
plt.plot(X_scaled, regressor.predict(X_scaled), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve) (scaled)
X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_scaled, y_scaled, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()