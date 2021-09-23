# Load model and predict
import sklearn
import pandas as pd
import numpy as np
from joblib import load
# Load trained model
model = load('linear.joblib')
while True:
    x=float(input("x:"))
    xs = np.array(x)
    prediction = model.predict(xs.reshape(1,-1))
    print(prediction)
    print("Press Ctrl + c to exit")
