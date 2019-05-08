#!/usr/bin/env python3

__author__ = "Yxzh"

import tensorflow as tf
from tensorflow import keras
import os
import gzip
import numpy as np

batch_size = 20

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def f_load_mnist(path, kind = 'train'):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
	images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
	
	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)
	
	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 28, 28)
	
	return images, labels

train_images, train_labels = f_load_mnist('data/fashion', kind = 'train')
test_images, test_labels = f_load_mnist('data/fashion', kind = 't10k')
train_images = train_images / 255.0  # 标准化
test_images = test_images / 255.0
model = keras.Sequential([  # 神经网络
	keras.layers.Flatten(input_shape = (28, 28)),  # 扁平化灰度值
	keras.layers.Dense(128, activation = tf.nn.relu),  # 128神经元全连接层
	keras.layers.Dense(10, activation = tf.nn.softmax)  # 输出层，维度10，和为1，为预测概率
])

model.compile(optimizer = tf.train.AdamOptimizer(),  # 喂损失函数，梯度下降逆向传输， 精准型
              loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 5)  # 训练，每Batch 60000组，50个Batch
print(train_images.shape, train_labels.shape)

test_loss, test_acc = model.evaluate(test_images, test_labels, 32, 1)
print('Test accuracy:', test_acc)
prediction = model.predict(train_images)
print(prediction[203])
print("the number 203 is", class_names[int(np.argmax(prediction[203]))])
