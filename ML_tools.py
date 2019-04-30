#!/usr/bin/env python3

__author__ = "Yxzh"

import tensorflow as tf
import numpy as np
import time


def f_import_CCPP(number):
	filenames = []
	for i in range(1, number + 1):
		filenames.append("CCPP/CCPP_%s.csv" % i)
		
	samples = np.zeros(shape = (4, 0), dtype = np.float64)
	labels = np.zeros(shape = (1, 0), dtype = np.float64)
	
	# csv读取初始化
	start_time = time.time()
	filename_queue = tf.train.string_input_producer(filenames, shuffle = False, capacity = 1)
	
	reader = tf.TextLineReader(skip_header_lines = 1)
	key, value = reader.read(filename_queue)
	record_defaults = [[.0], [.0], [.0], [.0], [.0]]
	AT, V, AP, RH, PE = tf.decode_csv(value, record_defaults = record_defaults)
	
	# 加载csv数据
	with tf.Session() as sess:
		coord = tf.train.Coordinator()  # 线程协调器
		threads = tf.train.start_queue_runners(coord = coord)  # 启动线程
		is_second_read = 0
		line1_name = b'%s:2' % b"CCPP/CCPP_1.csv"
		iteration = 1
		
		while True:  # 开始加载循环
			iteration += 1
			AT_, V_, AP_, RH_, PE_, line_key = sess.run([AT, V, AP, RH, PE, key])
			samples = np.c_[samples, [[AT_], [V_], [AP_], [RH_]]]
			labels = np.c_[labels, [PE_]]
			
			if iteration % 2000 == 0:
				print("Loading... (%s)" % line_key)
			
			if is_second_read == 0 and line_key == line1_name:  # 判断是否到文件尾
				is_second_read = 1
			elif is_second_read == 1 and line_key == line1_name:
				break
		
		coord.request_stop()  # 关闭线程
		coord.join(threads)
		sess.close()
	
	m = samples.shape[1]
	if m != labels.shape[1]:
		print("\033[1;31mDataset loading error.(The number of the samples and the labels are not same.)\033[0m")
		exit(1)
	print("Data loaded.({:n} data.)".format(m))
	print("\033[1;33m[{:n}s {:n}s/data]\033[0m\n--------------------------------------------------------".format(
		time.time() - start_time, (time.time() - start_time) / m))
	return samples, labels, m

def f_standardize(x):
	mean = x.mean()  # 计算平均数
	deviation = x.std()  # 计算标准差
	return ((x - mean) / deviation) , mean, deviation

def f_re_standradize(x, mean, deviation):
	return x * deviation + mean
