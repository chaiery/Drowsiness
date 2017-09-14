import tensorflow as tf
import numpy as np
import sys
import os
from flags import FLAGS
import tflearn_dev

_EPSILON = 1e-8


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list
	is over individual gradients. The inner list is over the gradient
	calculation for each tower.
	Returns:
	List of pairs of (gradient, variable) where the gradient has been averaged
	across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def _filter_prune_sparse(name, weights, random=False, **kwargs):
	weights_new = np.copy(weights)

	if random == True:
		weights = np.random.normal(0,1, weights.shape)

	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(weights), kwargs['percentage']*100)
	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(weights) < threshold
	weights_new[under_threshold] = 0
	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))
	return weights_new, ~under_threshold


def _filter_prune_n1(name, weights, random, **kwargs):
	weights_new = np.copy(weights)

	if random == True:
		weights = np.random.normal(0,1, weights.shape)

	_, _, num_channel, num_filter = weights.shape
	ls = []
	for channel in range(num_channel):
	    for out in range(num_filter):
	        weight = weights[:,:,channel, out]
	        l = np.linalg.norm(weight,ord=1)
	        ls.append(l)
	ls = np.array(ls).reshape(num_channel, num_filter)

	if 'percentage' in kwargs.keys():
		threshold = np.percentile(abs(ls), kwargs['percentage']*100)
	elif 'threshold' not in kwargs.keys():
		sys.exit('Error!')
	else:
		threshold = kwargs['threshold']

	under_threshold = abs(ls) < threshold
	under_threshold_elem = np.zeros(weights.shape, dtype=bool)
	under_threshold_elem[:,:,under_threshold] = True
	weights_new[under_threshold_elem] = 0

	drop_percent = 100*np.sum(under_threshold)/len(under_threshold.reshape(-1))

	print('{}: {:.2f} percent weights are dropped. Random = {}'.format(name, drop_percent, random))

	return weights_new, ~under_threshold_elem


def apply_pruning(layer_names, trained_path, model_id, tf_config):
	sess = tf.Session(config=tf_config)
	dict_widx = {}
	with sess.as_default():
		saver = tf.train.import_meta_graph(trained_path+'.meta')
		saver.restore(sess, trained_path)
		print('Model restored from ', trained_path)

		for i in range(len(layer_names)):
			var_name = layer_names[i]+'/kernel:0'
			var = [v for v in tf.global_variables() if v.name == var_name][0]
			weight = var.eval()
			weight, widx = _filter_prune_n1(layer_names[i], weight, random=False, percentage=0.3)

			dict_widx[var_name] = widx
			# Assign new value
			sess.run(var.assign(weight))

		#saver = tf.train.Saver(tf.global_variables())
		if not os.path.isdir(os.path.join(FLAGS.log_dir, 'prune_model')):
			os.mkdir(os.path.join(FLAGS.log_dir, 'prune_model'))
		checkpoint_path = os.path.join(FLAGS.log_dir, 'prune_model', '{}'.format(model_id))
		saver.save(sess, checkpoint_path)
	return dict_widx, checkpoint_path


def apply_prune_on_grads(grads_and_vars, dict_widx):
	for key, widx in dict_widx.items():
		count = 0
		for grad, var in grads_and_vars:
			if var.name == key:
				index = tf.cast(tf.constant(widx), tf.float32)
				grads_and_vars[count] = (tf.multiply(index, grad), var)
			count += 1
	return grads_and_vars


