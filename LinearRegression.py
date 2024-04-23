import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

dataset = pd.read_csv("datasets/house_prices.csv")
size=dataset['sqft_living']
price=dataset['price']
print(price)