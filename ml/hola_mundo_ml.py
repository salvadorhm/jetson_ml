# -*- coding: utf-8 -*-
"""hola_mundo_ml.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/salvadorhm/extraccion_conocimiento_bd/blob/main/ejercicios/hola_mundo_ml.ipynb

# Hola mundo de Machine Learning

## Importar las librerias
"""

from tensorflow import keras
import numpy as np
from time import time

print(keras.__version__) # Imprime la versión de keras
print(np.__version__) # Imprime la versión de numpy

"""# Defiir el modelo de ML

1. Se define un modelo de ML con 1 neurona (units=1) y 1 valor de entrada (x) input_shape=[1].

2. Se define la función de optimización "SGD" y la función para calcular la perdida "mean_squared_error".
"""

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss="mean_squared_error")

"""# Definir Xs y Ys

En este ejemplo los datos de muestra se optienen de la funcion y = x * 2
"""

xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=float)
ys = np.array([2.0,4.0,6.0,8.0,10.0,12.0], dtype=float)

"""# Entrenamiento

Se realiza el entrenamiento del modelo 500 epocas (veces), y en cada epoca se puede ver que el valor de perdida es más cercano a 0.
"""

tiempo_inicial = time() 
model.fit(xs,ys, epochs=5000)
tiempo_final = time() 
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Training time:{}".format(tiempo_ejecucion))

"""# Predicciones

Después de que se entreno el modelo, este ya se puede usar para realizar predicciones
"""

print(model.predict([10]))  # Valor esperado 20
print(model.predict([56]))  # Valor esperado 112
print(model.predict([45]))  # Valor esperado 90
print(model.predict([89]))  # Valor esperado 178