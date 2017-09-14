#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from datetime import datetime
import re
import time
import os
import time

## Import Customized Functions
import model
from data_loading import ReadData
from flags import FLAGS
import tflearn_dev as tflearn
from tensorflow.python.framework import ops

_FLOATX = tf.float32
_EPSILON = 1e-10

class Train():
	def __init__(self, model, img_size, learning_rate, run_id, config, set_id, train_range, vali_range, wd):
		self.model = model
		self.img_size = img_size
		self.learning_rate = learning_rate
		self.run_id = run_id
		self.tf_config = config
		self.set_id = set_id
		self.train_range = train_range
		self.vali_range = vali_range
		self.wd = wd


	def _build_graph(self):
		global_step = tf.contrib.framework.get_or_create_global_step()

		# Calculate logits using training data and vali data seperately
		logits = getattr(model, self.model)(self.batch_data, self.wd)
		

		with tf.name_scope("Crossentropy"):
			
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.batch_labels))
			weight_loss = tf.get_collection('losses')
			self.loss = loss
			if len(weight_loss)>0:
				self.total_loss = tf.add(tf.add_n(weight_loss), loss, name='total_loss')
			#self.total_loss = loss
			else:
				self.total_loss = loss

		with tf.name_scope("train"):
			#opt = tf.train.AdamOptimizer(self.learning_rate)
			learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                           100, 0.96, staircase=True)


			opt = tf.train.GradientDescentOptimizer(learning_rate)
			#opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
			#opt = tf.train.RMSPropOptimizer(self.learning_rate)
			
			grads_and_vars = opt.compute_gradients(self.total_loss)

			'''
			for grad, var in grads_and_vars:
				if grad is not None:
					tf.summary.histogram(var.op.name + '/gradients', grad)
			'''

			apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

			for var in tf.trainable_variables():
				tf.summary.histogram(var.op.name, var)
			

		with tf.name_scope("accuracy"):
			prob = tf.nn.softmax(logits)
			self.prediction = tf.argmax(prob,1)
			correct_prediction = tf.equal(self.prediction, tf.argmax(self.batch_labels,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.train_op = tf.group(apply_gradient_op)
		self.summary_op = tf.summary.merge_all()

		
	def train(self, **kwargs):
		ops.reset_default_graph()
		sess = tf.Session(config=self.tf_config)

		with sess.as_default():
			# Data Reading objects
			train_data = ReadData('train', self.img_size, self.set_id, self.train_range)
			vali_data = ReadData('validation', self.img_size, self.set_id, self.vali_range)

			train_batch_data, train_batch_labels = train_data.read_from_files()
			vali_batch_data, vali_batch_labels = vali_data.read_from_files()
			self.am_training = tf.placeholder(dtype=bool, shape=())
			self.batch_data = tf.cond(self.am_training, lambda:train_batch_data, lambda:vali_batch_data)
			self.batch_labels = tf.cond(self.am_training, lambda:train_batch_labels, lambda:vali_batch_labels)

			self._build_graph()
			self.saver = tf.train.Saver(tf.global_variables())
			# Build an initialization operation to run below
			init = tf.global_variables_initializer()
			sess.run(init)


			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			# This summary writer object helps write summaries on tensorboard
			summary_writer = tf.summary.FileWriter(FLAGS.log_dir+self.run_id)
			summary_writer.add_graph(sess.graph)

			train_error_list = []
			val_error_list = []

			print('Start training...')
			print('----------------------------------')


			train_steps_per_epoch = FLAGS.num_train_images//FLAGS.train_batch_size
			report_freq = train_steps_per_epoch

			train_steps = FLAGS.train_epoch * train_steps_per_epoch

			durations = []
			train_loss_list = []
			train_total_loss_list = []
			train_accuracy_list = []

			
			best_accuracy = 0

			for step in range(train_steps):
				tflearn.is_training(True)
				#print('{} step starts'.format(step))

				start_time = time.time()

				_, summary_str, loss_value, total_loss, accuracy = sess.run([self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
														feed_dict={self.am_training: True})
				#sess.run(self.batch_labels,feed_dict={self.am_training: True})

				duration = time.time() - start_time
				#print('{} step starts {}'.format(step, duration))
				
				durations.append(duration)
				train_loss_list.append(loss_value)
				train_total_loss_list.append(total_loss)
				train_accuracy_list.append(accuracy)

				assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
				
				
				if step%report_freq == 0:
					start_time = time.time()

					summary_writer.add_summary(summary_str, step)

					sec_per_report = np.sum(np.array(durations))
					train_loss = np.mean(np.array(train_loss_list))
					train_total_loss = np.mean(np.array(train_total_loss_list))
					train_accuracy_value = np.mean(np.array(train_accuracy_list))

					train_loss_list = []
					train_total_loss_list = []
					train_accuracy_list = []
					durations = []

					train_summ = tf.Summary()
					train_summ.value.add(tag="train_loss", simple_value=train_loss.astype(np.float))
					train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
					train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

					summary_writer.add_summary(train_summ, step)
                                                      
					vali_loss_value, vali_accuracy_value = self._full_validation(vali_data, sess)

					if vali_accuracy_value>best_accuracy:
						best_accuracy = vali_accuracy_value

						model_dir = os.path.join(FLAGS.log_dir, self.run_id, 'model')
						if not os.path.isdir(model_dir):
							os.mkdir(model_dir)
						checkpoint_path = os.path.join(model_dir, 'vali_{:.3f}'.format(vali_accuracy_value))

						self.saver.save(sess, checkpoint_path, global_step=step)


					vali_summ = tf.Summary()
					vali_summ.value.add(tag="vali_loss", simple_value=vali_loss_value.astype(np.float))
					vali_summ.value.add(tag="vali_accuracy", simple_value=vali_accuracy_value.astype(np.float))

					summary_writer.add_summary(vali_summ, step)
					summary_writer.flush()

					vali_duration = time.time() - start_time

					format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, accuracy = %.4f, vali_loss = %.4f, vali_accuracy = %.4f (%.3f ' 'sec/report)')
					print(format_str % (step//report_freq, train_loss, train_total_loss, train_accuracy_value, vali_loss_value, vali_accuracy_value, sec_per_report+vali_duration))
				

	def test(self):
		ops.reset_default_graph()
		test_data = ReadData(status='test', shape=self.img_size)
		test_batch_data, test_batch_labels = test_data.read_from_files()

		logits = getattr(Model, self.model)(test_batch_data)

		prob = tf.nn.softmax(logits)
		prediction = tf.argmax(prob,1)
		correct_prediction = tf.equal(prediction, tf.argmax(test_batch_labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		saver = tf.train.Saver(tf.all_variables())
		sess = tf.Session(config=self.tf_config)

		saver.restore(sess, FLAGS.test_ckpt_path)
		print('Model restored from ', FLAGS.test_ckpt_path)

		prediction_array = np.array([]).reshape(-1, FLAGS.num_categories)

		num_batches = FLAGS.num_test_images//FLAGS.test_batch_size
		accuracy_list = []

		for step in range(num_batches):
			
			batch_prediction_array, batch_accuracy = sess.run(
				[self.prediction, self.accuracy])
			prediction_array = np.concatenate((prediction_array, batch_prediction_array))
			accuracy_list.append(accuracy)

		accuracy = np.mean(np.array(accuracy_list, dtype=np.float32))

		return prediction_array, accuracy_list


	def _full_validation(self, vali_data, sess):
		tflearn.is_training(True)
		num_batches_vali = FLAGS.num_eval_images // FLAGS.train_batch_size

		loss_list = []
		accuracy_list = []

		for step_vali in range(num_batches_vali):
			loss, accuracy = sess.run([self.loss, self.accuracy], 
											feed_dict={self.am_training: False})
			
			loss_list.append(loss)
			accuracy_list.append(accuracy)


		vali_loss_value = np.mean(np.array(loss_list))
		vali_accuracy_value = np.mean(np.array(accuracy_list))

		return vali_loss_value, vali_accuracy_value


def main(argv=None):
	os.environ["CUDA_VISIBLE_DEVICES"]="2" 
	tf_config=tf.ConfigProto() 
	tf_config.gpu_options.allow_growth=True 
	#tf_config.gpu_options.per_process_gpu_memory_fraction=0.9
	img_size = (10,128,128,1)
	set_id = 'noglasses_glasses_3D_10_5_7'
	train_range = slice(0,10)
	vali_range = slice(11,14)

	model = 'baseline_3D'
	learning_rate = 1e-3
	weight_decay = 1e-5
	option = 1


	run_name = 'baseline'
	run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))


	train = Train(model, img_size, learning_rate, run_id, tf_config, set_id, train_range, vali_range, weight_decay)
	train.train()



if __name__ == '__main__':
	tf.app.run()
