# Multiple Linear Regression
# Linear Regression assumptions:
# 1) Linearity
# 2) Homoscedasticity
# 3) Multivariate normality
# 4) Independence of errors
# 5) Lack of multicollinearity

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# No missing values so can continue on

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap - never include all dummy variables in your columns
# Always omit 1 dummy variable! Remove the first column (index 0) as all information 
# is in the other 2 columns and you will be repeating a variable
X = X[:, 1:]

# Find shape of the DataFrame
X.shape

# Visualize the relationship between the features and the response using scatterplots
plt.figure()
sns.pairplot(dataset, x_vars=['R&D Spend','Administration','Marketing Spend','State'], y_vars='Profit', size=7, aspect=0.7)
plt.savefig("seaborn_pair_plot.png")

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - not needed for multiple linear regression as the library takes care of it for you
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set results and display the accuracy (RMSE)
y_pred = regressor.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Not try feature selection to improve the model
# Feature seletion - Choose independent variables & keep only import predictors
# Options: All-in, Backward Elimination, Forward Selection, Bidirectional Elimination and Score Comparison
# See pdf in folder to read about the options
# Here we weant to include them all

# Backward Elimination Feature selection
def backwardElimination(x, SL):

    # Firstly add a column of ones to add the intercept (b0 column) - add at the start of X
    x = np.append(arr = np.ones((len(x), 1)).astype(int), values = x, axis = 1)

    # Initiate variables
    numIterations = len(x[0])
    iv_indexes = range(0,len(x[0]))
    x_opt = x
    
    # For loop to iterate through predictors/regressors 
    for iv in range(0,numIterations):
        # Fit a model to all regressors using ordinary least squares algorithm
        regressor_OLS = sm.OLS(y, x_opt).fit()
        
        # Look for highest p value that is above the significance level and remove
        pvalues = regressor_OLS.pvalues
        if max(pvalues)>SL:
            # Get index of maximum value
            index_to_remove = pvalues.argmax(axis=0)
            
            # Remove index and keep looping
            del iv_indexes[index_to_remove]
            x_opt = x[:,iv_indexes] 
            
            # Print some things for guidance
            print('Remove this index:')
            print(index_to_remove)
        else: 
            print('Finishing loop')
            print(max(pvalues))
            break
        
    # Look at final summary
    print(regressor_OLS.summary())
    
    # Return
    return x_opt
    
  
# Run backwards elimination
SL = 0.05
X_SelectedFeatures = backwardElimination(X, SL) 

# Split into training and testing again
X_train, X_test, y_train, y_test = train_test_split(X_SelectedFeatures, y, test_size = 0.2, random_state = 0)

# Fit Multiple Linear Regression to the new training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set results and display the accuracy (RMSE)
y_pred = regressor.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



