import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('train_batch_size', 100,
	"""Train batch size""")

tf.app.flags.DEFINE_integer('test_batch_size', 256,
	"""Test batch size""")

tf.app.flags.DEFINE_integer('train_epoch', 100,
	"""Train epoch""")

tf.app.flags.DEFINE_string('log_dir', './log/',
	"""The diretory for logs""")

tf.app.flags.DEFINE_string('test_ckpt_path', 'tflearn_model',
	"""The path of the model to be tested""")

tf.app.flags.DEFINE_integer('crop_size', 110,
	"""Crop size""")

##########################################################################

#tf.app.flags.DEFINE_string('data_dir', './cifar-10-batches-bin',
#	"""The directory of dataset""")
tf.app.flags.DEFINE_string('data_dir', '/media/DensoML/DENSO ML/DrowsinessData',
	"""The directory of dataset""")

tf.app.flags.DEFINE_integer('num_categories', 2, #40000
	"""The number of categories""")

tf.app.flags.DEFINE_integer('num_train_images', 7000, #40000
	"""The number of images used for training""")

tf.app.flags.DEFINE_integer('num_eval_images', 4000, # 12000
	"""The number of images used for validation""")

tf.app.flags.DEFINE_integer('num_test_images', 50000,
	"""The number of images used for test""")

tf.app.flags.DEFINE_integer('label_bytes', 2,
	"""Label bytes""")

tf.app.flags.DEFINE_float('min_fraction_of_examples_in_queue', 0.2,
	"""Minimul fraction of examples in queue. Used for shuffling data""")


