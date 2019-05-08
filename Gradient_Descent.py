import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

np.set_printoptions(suppress = True)

# 训练样本
area = np.array([60, 76, 90, 92, 100, 105, 120, 129, 140, 174])
price = np.array([4700000, 5120000, 63800000, 6740000, 7160000, 7250000, 8990000, 9050000, 9800000, 10560000])

# 测试样本
test_area = np.array([51, 93, 124])
test_price = np.array([4230000, 6638000, 8720000])

# 模型
W = tf.Variable(25000, dtype = tf.float64)
T = tf.Variable(2, dtype = tf.float64)
x = tf.placeholder(tf.float64)
quadratic_model = W * x

# 期望 损失函数
expectation = tf.placeholder(np.float64)
squared_deltas = tf.abs(quadratic_model - expectation)
loss = tf.reduce_sum(squared_deltas) / (2 * area.size)
test_loss = tf.reduce_sum(squared_deltas) / (2 * test_area.size)

# 梯度下降
learning_rate = 5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.as_default()
sess.run(init)
scale = 95
loss_v = 0

# matplotlib 初始化
matplotlib.rcParams["figure.figsize"] = (10.8, 7.2)
ls = np.linspace(0, 180, 500)
plt.ylim(0, 12000000)
plt.xlim(0, 180)
plt.ticklabel_format(style = "plain")
plt.title("Regression Test @ " + "{:1.4f}".format(learning_rate) + " Learning Rate")
plt.plot(area, price, "rx", label = "Train")
plt.plot(ls, sess.run(quadratic_model, {x: ls}), "-", linewidth = 2.0, label = "Original")
n_pic = 1

# 训练
print("\033[1;31mOriginal loss = {0}\033[0m".format(sess.run(loss, {x: area, expectation: price})))
for i in range(scale * 5):
	sess.run(train, {x: area, expectation: price})
	loss_v = sess.run(loss, {x: area, expectation: price})
	print("i =", i, "\t\033[1;31mTrain loss =", loss_v, "\033[0m")
	if (i + 1) % scale == 0:
		plt.plot(ls, sess.run(quadratic_model, {x: ls}), "-", linewidth = 0.6,
		         label = str(n_pic * scale) + " iterations")
		n_pic += 1
print("(W, T)")
print("RESULT:", sess.run((W, T)))

# 保存模型
saver = tf.train.Saver()
saver.save(sess, "model/Regression.ckpt")

# 测试
plt.plot(test_area, test_price, "bo", label = "Test")
loss_test = sess.run(test_loss, {x: test_area, expectation: test_price})
print("Test Loss : {:e}\t".format(loss_test), "Train Loss: {:e}".format(loss_v))

# 收尾
plt.text(100, 700000, "Train Loss: {:.4e}\n Test Loss: {:.4e}".format(loss_v, loss_test), fontsize = "large", color = "blue")
plt.legend()
plt.show(size = (1080, 720))
