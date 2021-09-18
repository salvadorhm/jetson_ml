import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('model.h5')

x = float(input("X value:"))
y = model.predict([x])
print(y)