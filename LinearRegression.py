import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

#read csv

dataset = pd.read_csv("datasets/house_prices.csv")
size=dataset['sqft_living']
price=dataset['price']
print(price)

#machine learing handle arrays not dataframes
# we need array with single item
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

model = LinearRegression()
model.fit(x, y) # sklearn is going to train the model with help of fradient descent

#this is the b1
print(model.coef_[0])
#this is b0 in our model
print(model.intercept_[0])

#visualize the dataset with the fitted model
plt.scatter(x, y, color= 'green')
plt.plot(x, model.predict(x), color = 'black')
plt.title ("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()


#Predicting the prices
print("Prediction by the model:" , model.predict([[2000]])) #Prediction by the model: [[517666.39270042]]
