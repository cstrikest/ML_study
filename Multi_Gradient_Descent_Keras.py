#!/usr/bin/env python3

__author__ = "Yxzh"

import numpy as np
from tensorflow import keras
import pandas as pd

print("========================LOADING========================")
samples = np.array(pd.read_csv("data/CCPP/CCPP_train.csv", usecols = ["AT", "V", "AP", "RH"]))
labels = np.array(pd.read_csv("data/CCPP/CCPP_train.csv", usecols = ["PE"]))
test_samples = np.array(pd.read_csv("data/CCPP/CCPP_test.csv", usecols = ["AT", "V", "AP", "RH"]))
test_labels = np.array(pd.read_csv("data/CCPP/CCPP_test.csv", usecols = ["PE"]))
print("{} datas loaded.".format(samples.shape[0]))

model = keras.Sequential()
model.add(keras.layers.Dense(8192, activation = "relu", input_dim = 4, kernel_initializer = "normal"))
model.add(keras.layers.Dense(1024, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(128, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(4, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(1, kernel_initializer = "normal"))
model.compile(optimizer = "adam", loss = "mse")

print("========================TRAINING========================")
model.fit(samples, labels, 128, epochs = 20)

print("========================SAVING========================")
model.save("model/Multi_Gradient_Descent_Keras.h5")
print("Saved model to model/Multi_Gradient_Descent_Keras.h5")

print("========================TESTING========================")
print("The loss on test data:", model.evaluate(test_samples, test_labels, batch_size = 8))
print(model.predict(samples[:10]))