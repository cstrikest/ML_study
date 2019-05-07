#!/usr/bin/env python3

__author__ = "Yxzh"

import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd


samples = np.array(pd.read_csv("data/CCPP/CCPP_train.csv", usecols = ["AT", "V", "AP", "RH"]))
labels = np.array(pd.read_csv("data/CCPP/CCPP_train.csv", usecols = ["PE"]))

test_samples = np.array(pd.read_csv("data/CCPP/CCPP_test.csv", usecols = ["AT", "V", "AP", "RH"]))
test_labels = np.array(pd.read_csv("data/CCPP/CCPP_test.csv", usecols = ["PE"]))

print("{} datas loaded.".format(samples.shape[0]))

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation = "relu", input_dim = 4, kernel_initializer = "normal"))
model.add(keras.layers.Dense(12, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(8, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(4, activation = "relu", kernel_initializer = "normal"))
model.add(keras.layers.Dense(1, kernel_initializer = "normal"))

model.compile(optimizer = "adam", loss = "mse")

history = model.fit(samples, labels, 48, epochs = 30)

model.save("model/Multi_Gradient_Descent_Keras.h5")

result = model.evaluate(test_samples, test_labels, batch_size = 8)
#
# print(model.predict(np.array([
# 	[14.96, 41.76, 1024.07, 73.17, 463.26],
# 	[25.18, 62.96, 1020.04, 59.08, 444.37],
# 	[5.11, 39.4, 1012.16, 92.14, 488.56],
# 	[20.86, 57.32, 1010.24, 76.64, 446.48],
# 	[10.82, 37.5, 1009.23, 96.62, 473.9],
# 	[26.27, 59.44, 1012.23, 58.77, 443.67],
# 	[15.89, 43.96, 1014 .02, 75.24, 467.35],
# 	[9.48, 44.71, 1019.12, 66.43, 478.42],
# 	[14.64, 45, 1021.78, 41.25, 475.98]
# ])))
print(result)

print(model.predict(samples[:10]))