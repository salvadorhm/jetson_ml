import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from time import time

dataframe = pd.read_csv('datos00.csv')
df_x = dataframe[['x']]
df_y = dataframe['y']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=1234)

model = AutoSklearnRegressor(time_left_for_this_task=100, per_run_time_limit=None, n_jobs=1)

i = time()
print("Init training")
model.fit(x_train,y_train,x_test,y_test)
print("finish")
e = time()
t = e - i
print("training time {}".format(t))
print(model.sprint_statistics())

# Evaluamos el mejor modelo
y_hat = model.predict(x_test)
acc = r2_score (y_test, y_hat)
print("Accuracy: %.3f" % acc)
print(model.show_models())
# print(model.get_models_with_weights())

# print(model.cv_results_)

# Dump train model to joblib
dump(model,'linear.joblib')
predictions = model.predict(x_test)
print(predictions)
