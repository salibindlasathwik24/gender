
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data.csv")

X = df[['male', 'female']]
y = df["ratio"]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained successfully")
