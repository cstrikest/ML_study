#!/usr/bin/env python3

__author__ = "Yxzh"

import numpy as np
import SimpleNN as SNN
import time


sample = np.array(
	[[0.79, 0], [0, 1], [1, 2.34], [1.23, 0.8592], [1.92, 3.086], [3.1, 0.5], [3.1, 3.6], [4.5, 5.9], [6.5, 5.07],
	 [5.125, 6.951], [7.44, 4.61], [7.7, 7.8], [6.9, 5.84], [6.2, 8.63]]).T
label = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


S = SNN.SimpleNN(8, sample, label)


for _ in range(0, 1000):
	S.forward()
	S.backward()
print(S.A2)

print(S.test(np.array([[1, 1]]).T))
