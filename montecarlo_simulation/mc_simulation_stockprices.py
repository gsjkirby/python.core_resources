#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:56:00 2018

@author: georginakirby

Tutorial: https://www.youtube.com/watch?v=_T0l015ecK4

"""

# Import packages
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like # to solve the import error for the line below
import pandas_datareader.data as web # conda install -c anaconda pandas-datareader
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Define start and end dates
start = dt.datetime(2017,01,03)
end = dt.datetime(2017,11,20)

# Read in pricing
prices = web.DataReader('AAPL', 'google', start, end)['Close']

# Returns 
returns = prices.pct_change()

# Last price is the last index
last_price = prices[-1]

# Number of Simulations
num_simulations = 1000

# Number of Days to predict in the future
num_days = 252

# Create a data frame for the simulations
simulation_df = pd.DataFrame()

# Loop through the simulations - break forloop once all days are accounted for
for x in range(num_simulations):
    count = 0
    
    # Daily standard deviation
    daily_vol = returns.std()
    
    # Create empty list for all prices
    price_series = []
    
    # First price = last price * (1 + random price based on volatility distribution)
    price = last_price * (1 + np.random.normal(0, daily_vol))
    
    # Append price to the list
    price_series.append(price)
    
    # Check to see if we need to break the forloop
    for y in range(num_days):
        if count == 251:
            break
        
        # Otherwise, reference individual element in the list and multiply 
        price = price_series[count] * (1 + np.random.normal(0, daily_vol))
        
        # Append price to the price series
        price_series.append(price)
        
        # Incrememnt count
        count += 1
    
    simulation_df[x] = price_series

# Visualise the simulation
fig = plt.figure()
fig.suptitle('Monte Carlo Simulation: AAPL')
plt.plot(simulation_df)
plt.axhline(y = last_price, color = 'r', linestyle = '-')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

