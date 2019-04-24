#!/usr/bin/env python3

__author__ = "Yxzh"

import numpy as np


class SimpleNN(object):
	def __init__(self, neurons, sample, label):
		"""
		初始化神经网络
		:param neurons: 隐藏层神经元数量
		:param sample: 训练样本
		:param label: 训练标签
		"""
		#  输入层
		self.X = sample
		self.Y = label
		self.m = sample.shape[1]
		self.neurons = neurons
		
		#  隐藏层
		self.W1 = np.random.randn(neurons, sample.shape[0]) * 0.01  # 权重矩阵
		self.B1 = np.zeros((neurons, 1))  # 偏移矩阵
		
		#  输出层
		self.W2 = np.random.randn(1, neurons) * 0.01  # 权重矩阵
		self.B2 = 0  # 偏移矩阵
	
	# 各种激活函数和导数
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def tanh(self, z):
		return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
	
	def d_tanh(self, a):
		return 1 - np.square(a)
		
	# 前向传播
	def forward(self):
		self.Z1 = np.matmul(self.W1, self.X) + self.B1  # 第一层预测函数
		self.A1 = self.tanh(self.Z1)  # 激活函数
		self.Z2 = np.matmul(self.W2, self.A1) + self.B2  # 第二层预测函数
		self.A2 = self.sigmoid(self.Z2)  # 激活函数
		return self.A2
	
	# 反向传播
	def backward(self):
		self.dZ2 = self.A2 - self.Y  # 损失
		self.dW2 = (1 / self.m) * np.matmul(self.dZ2, self.A1.T)  # 逆向求导
		self.dB2 = (1 / self.m) * np.sum(self.dZ2, axis = 1, keepdims = True)
		self.dZ1 = np.matmul(self.W2.T, self.dZ2) * self.d_tanh(self.Z1)
		self.dW1 = (1 / self.m) * np.matmul(self.dZ1, self.X.T)
		self.dB1 = (1 / self.m) * np.sum(self.dZ1, axis = 1, keepdims = True)
		
		self.W1 = self.W1 - 0.08 * self.dW1  # 权重反馈
		self.B1 = self.B1 - 0.08 * self.dB1
		self.W2 = self.W2 - 0.08 * self.dW2
		self.B2 = self.B2 - 0.08 * self.dB2
	
	def test(self, test):
		self.test = test
		self.Z1 = np.matmul(self.W1, self.test) + self.B1
		self.A1 = self.tanh(self.Z1)
		self.Z2 = np.matmul(self.W2, self.A1) + self.B2
		self.A2 = self.sigmoid(self.Z2)
		return self.A2
