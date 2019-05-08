#!/usr/bin/env python3

__author__ = "Yxzh"

import numpy as np
from tensorflow import keras
import pandas as pd


csv_loader = pd.read_csv("data/housing.csv", delim_whitespace = True, header = None)
samples = np.array(csv_loader)[:, 0: 13]
labels = np.array(csv_loader)[:, 13: 14]

print(samples, labels)
model = keras.Sequential()
model.add(keras.layers.Dense(13, input_dim = 13, kernel_initializer = 'normal', activation = 'relu'))
model.add(keras.layers.Dense(7, kernel_initializer = 'normal', activation = 'relu'))
model.add(keras.layers.Dense(3, kernel_initializer = 'normal', activation = 'relu'))
model.add(keras.layers.Dense(1, kernel_initializer = 'normal'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(samples, labels, 4, epochs = 200)

model.save("model/housing.h5")

print(
	model.predict(np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]])))
print(
	model.predict(np.array([[0.02731, 0.00, 7.070, 0, 0.4690, 6.4210, 78.90, 4.9671, 2, 242.0, 17.80, 396.90, 9.14]])))
print(
	model.predict(np.array([[0.02729, 0.00, 7.070, 0, 0.4690, 7.1850, 61.10, 4.9671, 2, 242.0, 17.80, 392.83, 4.03]]))
)
