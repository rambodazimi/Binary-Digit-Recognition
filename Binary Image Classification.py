import numpy as np
import tensorflow as tf
from autils import *

"""
Author: Rambod Azimi
This Python code constructs a 4 layer neural network to recognize two handwritten digits, zero and one.
The data set contains 1000 training examples of written digits.
Each training example is a 20 pixel by 20 pixel grayscale image of the digit.
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
"""

# loading the data from autils file into 2 variables X_train (inputs) and Y_train (outputs)
X_train, Y_train = load_data()

# constructing the neural network with 2 hidden layers and one output layer with 25, 15, and 1 neurons respectively, using Sequential function in TensorFlow

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(400,)),
    tf.keras.layers.Dense(units=25, activation='sigmoid', name='layer1'),
    tf.keras.layers.Dense(units=15, activation='sigmoid', name='layer2'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='layer3')
])

# summary of the neural network
model.summary()

# loading the layers of the model into layer1, layer2, and layer3
[layer1, layer2, layer3] = model.layers

# loading the parameters W and b into their corresponding variables
W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()

# compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy())

# fitting the model
model.fit(X_train, Y_train, epochs=20)

# prediction and testing
prediction1 = model.predict(X_train[0].reshape(1,400))  # a zero
if(prediction1 >= 0.5):
    print("1")
else:
    print("0")

prediction2 = model.predict(X_train[500].reshape(1,400))  # a one
if(prediction2 >= 0.5):
    print("1")
else:
    print("0")

