"""A very basic intro to map, filter and reduce in python.
See video here:
https://www.youtube.com/watch?v=hUes6y2b--0
"""

import math
import statistics

# Creating a map - single numerical values
def area(r):
    """Function to define area of a circle with radius r"""
    return math.pi * (r**2)

radii = [2, 5, 7.1, 0.3, 10]
map(area,radii)


# Creating a map - using tuples
temps = [("Berlin", 23), ("London", 19), ("Paris", 27)]
celc_to_faren = lambda data: (data[0], (9/5)*data[1]+ 32)
map(celc_to_faren,temps)


# Creating a filter - single numerical values
data = [0.3, 2.8, 4.1, 4.7, -0.1, 3.0]
avg = statistics.mean(data)
filter(lambda x: x>avg, data)

# Creating a filter - removing missing values
countries = ["England", "", "Spain", "Brazil", "", "", "France"]
filter(None, countries)


# Creating a reduce
# 99% of the time a simple for loop is more readable so use this instead
