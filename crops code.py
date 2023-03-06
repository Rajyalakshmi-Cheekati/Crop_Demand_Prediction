
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_excel("tomato_data.xlsx")
prices = []
for i in df["Today price"].tolist():
  price = i.split()
  prices.append(price[0])
supply = df["Today arrival"].tolist()
supply_lm = df["Last month arrival"].tolist()
supply_lw = df["Last week arrival"].tolist()

df = pd.read_excel("tomato_data.xlsx", "FEB 2020")
for i in df["Today price"].tolist():
  price = i.split()
  prices.append(price[0])
supply = supply + df["Today arrival"].tolist()
supply_lm = supply_lm + df["Last month arrival"].tolist()
supply_lw = supply_lw + df["Last week arrival"].tolist()

df = pd.read_excel("tomato_data.xlsx", "MAR 2020")
for i in df["Today price"].tolist():
  price = i.split()
  prices.append(price[0])

supply = supply + df["Today arrival"].tolist()
supply_lw = supply_lw + df["Last week arrival"].tolist()
supply_lm = supply_lm + df["Last month arrival"].tolist()
dates = list(range(31+29+31))

new_supply_lm = []
for i in range(len(supply_lm)):
  new_supply_lm.append(float(supply_lm[i])**3)

plt.scatter(new_supply_lm, prices)
plt.xlabel("Supply Quantity")
plt.ylabel("Price Demand")

regr = LinearRegression()

X = np.array(new_supply_lm).reshape(-1, 1)
Y = np.array(prices).reshape(-1, 1)

regr.fit(X, Y)
print(regr.score(X, Y))

y_pred = regr.predict(X)
plt.plot(new_supply_lm, y_pred, color ='k')

plt.scatter(new_supply_lm, prices, color ='b')
