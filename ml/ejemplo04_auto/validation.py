# Load model and predict
import sklearn
import pandas as pd
from joblib import load
# Load trained model
model = load('linear.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['x']]
predictions = model.predict(xs)
print(predictions)
