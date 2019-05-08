import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


# 样本与标签
sample = np.zeros(shape = (5, 0), dtype = np.float64)
label = np.zeros(shape = (1, 0), dtype = np.float64)

# csv读取初始化
start_time = time.time()
filename_queue = tf.train.string_input_producer(["data/CCPP/CCPP_train.csv"], shuffle = False, capacity = 1)

reader = tf.TextLineReader(skip_header_lines = 1)
key, value = reader.read(filename_queue)
record_defaults = [[.0], [.0], [.0], [.0], [.0]]
AT, V, AP, RH, PE = tf.decode_csv(value, record_defaults = record_defaults)

# 加载csv数据
with tf.Session() as sess:
	# 线程协调器
	coord = tf.train.Coordinator()
	
	# 启动线程
	threads = tf.train.start_queue_runners(coord = coord)
	is_second_read = 0
	line1_name = b'%s:2' % b"data/CCPP/CCPP_train.csv"
	iteration = 1
	
	# 开始加载循环
	while True:
		iteration += 1
		AT_, V_, AP_, RH_, PE_, line_key = sess.run([AT, V, AP, RH, PE, key])
		sample = np.c_[sample, [1, AT_, V_, AP_, RH_]]
		label = np.c_[label, [PE_]]
		
		if iteration % 1000 == 0:
			print("Loading... (%s)" % line_key)
		
		# 判断是否到文件尾
		if is_second_read == 0 and line_key == line1_name:
			is_second_read = 1
		elif is_second_read == 1 and line_key == line1_name:
			break
	
	# 关闭线程
	coord.request_stop()
	coord.join(threads)
	sess.close()

# 整理数据
m = sample.shape[1]
if m != label.shape[1]:
	print("\033[1;31mDataset loading error.(The number of the samples and the labels are not same.)\033[0m")
	exit(1)
print("Data loaded.({:n} data.)".format(m))
print("\033[1;33m[{:n}s {:n}s/data]\033[0m\n--------------------------------------------------------".format(
	time.time() - start_time, (time.time() - start_time) / m))

# 特征缩放 xn_scale = (xn - average(xn)) / standard(xn)
sample_scaling = np.ones(shape = (5, m), dtype = np.float64)
average = np.zeros(shape = (5, 1), dtype = np.float64)
standard = np.zeros(shape = (5, 1), dtype = np.float64)
for i in range(1, 5):
	average[i] = np.mean(sample[i])
	standard[i] = np.std(sample[i])
	for j in range(0, m - 1):
		sample_scaling[i, j] = (sample[i, j] - average[i]) / standard[i]

# 预测函数
t = tf.Variable(tf.zeros(shape = [1, 5], dtype = tf.float64), dtype = tf.float64)
x = tf.placeholder(tf.float64)
h = tf.matmul(t, x)

# 损失函数
y_label = tf.placeholder(tf.float64)
loss = tf.reduce_sum(tf.square(h - y_label)) / (2 * m)

# 梯度下降
learning_rate = 1.7e-2
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

# 初始化
sess = tf.Session()
sess.as_default()
init = tf.global_variables_initializer()
sess.run(init)
iteration = 5000
step = 25  # 1为不输出节 单步时间默认为每次迭代

# matplotlib 初始化
ls = [0]
loss_draw = [sess.run(loss, {x: sample_scaling, y_label: label})]
loss_temp = 0
t_0_draw = [0]
t_1_draw = [0]
t_2_draw = [0]
t_3_draw = [0]
t_4_draw = [0]
plt.ylim(-15, 7)
plt.xlim(0, iteration)
plt.ticklabel_format(style = "plain")
plt.title("Multi-Regression Test @ " + "{:e}".format(learning_rate) + " Learning Rate")

# 训练
start_time = time.time()
print("Training...")
for epoch in range(1, iteration + 1):
	sess.run(train, {x: sample_scaling, y_label: label})
	
	# 输出节
	if step != 1 and epoch % step == 0:
		loss_temp = sess.run(loss, {x: sample_scaling, y_label: label})
		print("epoch = {:d}\t\tloss = \033[1;31m{:e}\033[0m \033[1;33m[{:n}s]\033[0m".format(epoch, loss_temp,
		                                                                                     time.time() - start_time))
		[[t_0, t_1, t_2, t_3, t_4]] = sess.run(t)
		t_0_draw.append(t_0)
		t_1_draw.append(t_1)
		t_2_draw.append(t_2)
		t_3_draw.append(t_3)
		t_4_draw.append(t_4)
		ls.append(epoch)

# 输出结果 保存模型
print("\033[1;33m[{:n}s ({:n}s/step)]\033[0m\n--------------------------------------------------------".format(
	time.time() - start_time, (time.time() - start_time) / (iteration / step)))
plt.plot(ls, t_0_draw, "-", color = "red", label = "t_0", linewidth = 1)
plt.plot(ls, t_1_draw, "-", color = "green", label = "t_1", linewidth = 1)
plt.plot(ls, t_2_draw, "-", color = "blue", label = "t_2", linewidth = 1)
plt.plot(ls, t_3_draw, "-", color = "orange", label = "t_3", linewidth = 1)
plt.plot(ls, t_4_draw, "-", color = "purple", label = "t_4", linewidth = 1)
plt.legend()
plt.show()
print("t:", sess.run(t))
saver = tf.train.Saver()
saver.save(sess, "model/Multi-Regression.ckpt")

# 将测试数据缩放
def scale(n = []):
	for i in range(1, 5):
		n[i] = (n[i] - average[i]) / standard[i]
	return n

print("test:", sess.run(h, {x: np.array([scale([1, 25.18, 62.96, 1020.04, 59.08])]).T}))
