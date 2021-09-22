import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('model.h5')
while True:
    x = float(input("X value:"))
    y = model.predict([x])
    print(y)
    print("Press Ctrl + c to finish")