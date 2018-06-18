# Data Preprocessing Template

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Create independent variable matrix of features (choose every column except the last)
X = dataset.iloc[:, :-1].values

# Create dependent variable vector (3rd column)
y = dataset.iloc[:, 3].values

# Impute missing data (use Command+I to inspect the Imputer class)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# Fit the imputer to the 2nd and 3rd columns (index 1 and 2) in X 
# (the final index is not inclusive so it doesnt include index 3)
imputer = imputer.fit(X[:, 1:3])

# Now replace the missing data in X with the mean using the transform method 
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode the categorical variables in the independent variable (X) - the first column (index 0)
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# Onehotencode the categorical variable in X so that the algorithm knows it's 
# categorical and not to fit to the values
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encode the categorical dependent variable (y) - no need to onehotencode
# because there is only one variable so the algorithm will know it is categorical
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling for X - could scale by standardise (x-mean/std) or normalise (x-min/max-min)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# Apply scaling to test set (already fitted to training set X and need to be scaled on same basis)
X_test = sc_X.transform(X_test)

# Feature Scaling for y - fit and transform 
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)







