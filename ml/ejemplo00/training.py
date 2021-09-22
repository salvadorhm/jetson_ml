# -*- coding: utf-8 -*-
"""
## Importar las librerias
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from time import time

print(keras.__version__) # Imprime la versión de keras
print(np.__version__) # Imprime la versión de numpy

"""# Defiir el modelo de ML

1. Se define un modelo de ML con 1 neurona (units=1) y 1 valor de entrada (x) input_shape=[1].

2. Se define la función de optimización "SGD" y la función para calcular la perdida "mean_squared_error".
"""

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss="mean_squared_error", metrics=["accuracy"])

"""# Definir Xs y Ys

En este ejemplo los datos de muestra se optienen de la funcion y = x * 2
"""

xs = np.genfromtxt('datos0.csv', delimiter=',', skip_header=1 , usecols=0,dtype=float)
ys = np.genfromtxt('datos0.csv', delimiter=',', skip_header=1 , usecols=1,dtype=float)

x_training, x_test = xs[:8], xs[8:]
y_training, y_test = ys[:8], ys[8:]

print(x_training)
print(x_test)

plt.plot(xs,ys)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Demo')
plt.savefig('foo.png')
plt.show()

"""# Entrenamiento

Se realiza el entrenamiento del modelo 500 epocas (veces), y en cada epoca se puede ver que el valor de perdida es más cercano a 0.
"""

tiempo_inicial = time() 
print("Init Training")
model.fit(x_training,y_training, epochs=5000,verbose=0,workers=10,use_multiprocessing=True)
print("End Training")
tiempo_final = time() 
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Training time:{}".format(tiempo_ejecucion))

"""# Predicciones
Después de que se entreno el modelo, guardamos el modelo para usarlo despues
"""

results = model.evaluate(x_test, y_test,verbose=1,return_dict=True)
print("test accuracy:{}".format(results))

model.save("model.h5")
