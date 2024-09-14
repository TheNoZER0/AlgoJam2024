import math
import pandas as pd

# Load Thrifted Jeans data
jeans = pd.read_csv('Data Analysis/Thrifted Jeans Data.csv')

# plot the data
jeans.plot.scatter(x='Price', y='Resale Price', title='Resale Price vs. Price')
