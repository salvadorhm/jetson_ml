# Load model and predict
import sklearn
import pandas as pd
from joblib import load
# Load trained model
model = load('linear.joblib')
while True:
    xs = []
    x = float(input("x:"))
    xs.append([x])
    prediction = model.predict(xs)
    print(prediction)
    print("Press Ctrl + c to exit")
