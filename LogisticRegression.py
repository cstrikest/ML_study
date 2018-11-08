import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 模型
W = tf.Variable(2, dtype = tf.float32)
b = tf.Variable(4, dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sigmoid_model = 1 / (1 + np.power(np.e, (-1 * linear_model)))  # sigmund

# 期望 损失函数
yi = tf.placeholder(np.float32)
loss_square = -1 * tf.reduce_sum(yi * tf.log(sigmoid_model) + (1 - yi) * tf.log(1 - sigmoid_model)) / 8
# loss_square = tf.square(loss)

# 梯度下降
learning_rate = 0.000001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss_square)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.as_default()
sess.run(init)

# matplotlib 初始化
ls = np.linspace(0, 9, 30)
plt.title("Logistic Regression Test @ " + str(learning_rate) + " Learning Rate")
plt.plot([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1], "rx", label = "Data")
plt.plot(ls, sess.run(linear_model, {x: ls}), "-", linewidth = 2.0, label = "Original")
plt.ylim([-4, 4])
n_pic = 1
scale = 300

# 跑
for i in range(scale * 5):
	sess.run(train, {x: [1, 2, 3, 4, 5, 6, 7, 8], yi: [0, 0, 0, 0, 1, 1, 1, 1]})
	print("i =", i, "fx([1, 2, 3, 4, 5, 6, 7, 8]) =", sess.run(sigmoid_model, {x: [1, 2, 3, 4, 5, 6, 7, 8]}),
	      "loss_square =",
	      sess.run(loss_square, {x: [1, 2, 3, 4, 5, 6, 7, 8], yi: [0, 0, 0, 0, 1, 1, 1, 1]}), sess.run((W, b)))
	if (i + 1) % scale == 0:
		plt.plot(ls, sess.run(linear_model, {x: ls}), "-", linewidth = 0.6, label = str(n_pic * scale) + " iterations")
		n_pic += 1
print("RESULT:", sess.run((W, b)))
test = 2
result = sess.run(sigmoid_model, {x: test})
print(result)
plt.legend()
plt.show()
