import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


#  样本与标签
sample = np.array(
	[[0.79, 0], [0, 1], [1, 2.34], [1.23, 0.8592], [1.92, 3.086], [3.1, 0.5], [3.1, 3.6], [4.5, 5.9], [6.5, 5.07],
	 [5.125, 6.951], [7.44, 4.61], [7.7, 7.8], [6.9, 5.84], [6.2, 8.63]])
label = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
sample = np.insert(sample, 0, np.ones((1, 14)), 1).T

#  预测函数 (sigmoid included)
t = tf.Variable(tf.ones(shape = [1, 3], dtype = tf.float64), dtype = tf.float64)
x = tf.placeholder(dtype = np.float64)
h = 1 / (1 + tf.exp(tf.matmul(t, x)))

#  损失函数
y_label = tf.placeholder(dtype = np.float64)
loss = tf.reduce_sum(-y_label * tf.log(h) - (1 - y_label) * tf.log(1 - h)) * 1 / 14

#  梯度下降
learning_rate = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

#  初始化
sess = tf.Session()
sess.as_default()
init = tf.global_variables_initializer()
sess.run(init)
iteration = 10000
step = 1000  # 0为不输出节 单步时间默认为每次迭代

#  matplotlib 初始化
plt.ticklabel_format(style = "plain")
plt.title("LogisticRegression Test @ " + "{:e}".format(learning_rate) + " Learning Rate")
plt.plot(sample[1, : 7], sample[2, : 7], "rx", label = "sample 1")
plt.plot(sample[1, 7: 14], sample[2, 7: 14], "bx", label = "sample 2")

#  训练
start_time = time.time()
print("Training...")
for epoch in range(1, iteration + 1):
	sess.run(train, {x: sample, y_label: label})
	
	# 输出节
	if step != 0:
		if epoch % step == 0:
			loss_temp = sess.run(loss, {x: sample, y_label: label})
			print("epoch = {:d}\t\tloss = \033[1;31m{:e}\033[0m \033[1;33m[{:n}s]\033[0m".format(epoch, loss_temp,
			                                                                                     time.time() - start_time))

#  输出结果
print("\033[1;33m[{:n}s ({:n}s/step)]\033[0m\n--------------------------------------------------------".format(
	time.time() - start_time, (time.time() - start_time) / (iteration / (step if step != 0 else 1))))
t_result = sess.run(t)
print("t1: {0}\nt2: {1}\nt3: {2}\n--------------------------------------------------------".format(t_result[:, 0],
                                                                                                   t_result[:, 1],
                                                                                                   t_result[:, 2]))

#  测试
test = np.array([[1, 2, 2], [1, 6, 6], [1, 5, 5], [1, 4.5, 4.5], [1, 4, 4]], dtype = np.float64).T
result = sess.run(h, {x: test})
print("Test model by\n", test)
print("Result:", result)

plt.plot(test[1, 0], test[2, 0], "ro", label = "test 1")
plt.plot(test[1, 1], test[2, 1], "bo", label = "test 2")
plt.plot(test[1, 2], test[2, 2], "bo", label = "test 3")
plt.plot(test[1, 3], test[2, 3], "bo", label = "test 4")
plt.plot(test[1, 4], test[2, 4], "ro", label = "test 5")


plt.text(2 + 0.1, 2 + 0.1, "{:.2f}%".format((1 - result[0][0]) * 100))
plt.text(6 + 0.1, 6 + 0.1, "{:.2f}%".format(result[0][1] * 100))
plt.text(5 + 0.1, 5 + 0.1, "{:.2f}%".format(result[0][2] * 100))
plt.text(4.5 + 0.1, 4.5 + 0.1, "{:.2f}%".format(result[0][3] * 100))
plt.text(4 + 0.1, 4 + 0.1, "{:.2f}%".format((1 - result[0][4]) * 100))

plt.legend()
plt.show()

saver = tf.train.Saver()
saver.save(sess, "model/LogisticRegression.ckpt")
sess.close()
