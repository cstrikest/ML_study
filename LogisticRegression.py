import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


#  Sigmoid函数
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

#  样本与标签
sample = np.array([[1, 0], [0, 1], [1, 2], [1, 1], [2, 3], [3, 1], [3, 3], [4, 6], [6, 5], [5, 7], [7, 6], [7, 7], [8, 5], [6, 8]])
label = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
sample = np.insert(sample, 0, np.ones((1, 14)), 1)

#  预测函数
t = tf.Variable(tf.ones(shape = [1, 3], dtype = tf.float64), dtype = tf.float64)
x = tf.placeholder(dtype = np.float64)
h = 1 / (1 + tf.exp(tf.matmul(t, x)))

#  损失函数 ***此处预测函数有问题，其值永远为NAN。算式应该没有问题，是tf处理的问题。待解决。
y_label = tf.placeholder(dtype = np.float64)
loss = tf.reduce_sum(np.multiply(-y_label, tf.log(h)) - np.multiply((1 - y_label), tf.log(1 - x))) * -1 / 14

#  梯度下降
learning_rate = 1e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

# 初始化
sess = tf.Session()
sess.as_default()
init = tf.global_variables_initializer()
sess.run(init)
iteration = 500
step = 1  # 1为不输出节 单步时间默认为每次迭代

# 训练
start_time = time.time()
print("Training...")
print(sample)
print(label)
for epoch in range(1, iteration + 1):
	sess.run(train, {x: sample.T, y_label: label})
	
	# 输出节
	if step != 0 and epoch % step == 0:
		loss_temp = sess.run(loss, {x: sample.T, y_label: label})
		print("epoch = {:d}\t\tloss = \033[1;31m{:e}\033[0m \033[1;33m[{:n}s]\033[0m".format(epoch, loss_temp,
		                                                                                     time.time() - start_time))

# 输出结果 保存模型
print("\033[1;33m[{:n}s ({:n}s/step)]\033[0m\n--------------------------------------------------------".format(
	time.time() - start_time, (time.time() - start_time) / (iteration / step)))
saver = tf.train.Saver()
saver.save(sess, "model/LogisticRegression.ckpt")