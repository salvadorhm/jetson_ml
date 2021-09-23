import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import dump, load

dataframe = pd.read_csv('datos00.csv')
df_x = dataframe[['x']]
df_y = dataframe['y']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

y_hat = model.predict(x_test)
acc = r2_score (y_test, y_hat)
print("Accuracy: %.3f" % acc)
# Dump train model to joblib
dump(model,'linear.joblib')
predictions = model.predict(x_test)
print(predictions)
