"""Train a small linear-regression house-price baseline from a CSV file."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", type=Path, help="CSV containing area, bedrooms, bathrooms, and price columns")
args = parser.parse_args()

data = pd.read_csv(args.csv)
print(data.head())
X = data[['area', 'bedrooms', 'bathrooms']]   # Input features
y = data['price']                             # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#Always the same split,consistent results,easier to debug and explain.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Selling Price Predictions")
plt.show()
