from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange
import os
import numpy as np
import tensorflow as tf
from flags import FLAGS
import random
import scipy.ndimage
import sys
import glob
import pickle


SCENARIOS = ['noglasses', 'glasses', 'night_noglasses', 'night_glasses', 'sunglasses']
TRAIN = ['001', '002', '005', '006', '008', '009','012', '013', '015', \
		'020', '023', '024','031', '032', '033']


def get_data(set_id, seq_length, stride_frame, stride_seq, subject_index=TRAIN, scenarios=SCENARIOS):
	path = os.path.join(FLAGS.data_dir, set_id)

	if os.path.isdir(path):
		sys.exit('The directory is already exist')
	else:
		os.makedirs(path)

	# Build the prefix where we will read the data
	# The form is {subject_index}_{scenerio}
	r1 = [item for item in subject_index for i in range(len(scenarios))]
	r2 = scenarios*len(subject_index)
	files_pf = ['{}_{}'.format(a, b) for a, b in zip(r1, r2)] 

	#collection = np.array([], dtype='float32')
	log_f = open(os.path.join(FLAGS.data_dir, set_id+'_info'), 'a')

	for pf in files_pf:
		total = 0
		# Grab all files with the prefix
		files = glob.glob(FLAGS.data_dir + '/raw_data/' + pf + '*')
		for each_file in files:
			collection = []
			fname = each_file.split('/')[-1]
			with open(each_file, 'rb') as rf:
				imgs, annots = pickle.load(rf)

				for index in range (0,len(imgs)-seq_length*stride_frame,stride_seq):
				    # IF 2D input
					if seq_length == 1:
						data_point = imgs[index,:,:]
						label = int(annots[index])
						if label==1:
							label_array = np.array([1,0])
						elif label==0:
							label_array = np.array([0,1])
						#label_array = np.array([label, 1-label])
						row = np.concatenate((label_array, data_point.flatten()), axis=0)
						row = row.astype(np.uint8)

					# IF 3D input
					else:
						data_point = imgs[index:index+seq_length*stride_frame:stride_frame,:,:]
						label = annots[index:index+seq_length*stride_frame:stride_frame]

						if np.array(list(label),dtype='int32').sum()>seq_length*0.5:
							label_array = np.array([1,0])
						else:
							label_array = np.array([0,1])
						row = np.concatenate((label_array, data_point.flatten()), axis=0)
						row = row.astype(np.uint8)
					collection.append(row)

			rf.close()
			print('Finish extracting data from %s'%(each_file))

			samples_length = len(collection)
			total += samples_length
			collection = np.concatenate(collection, axis=0)
			# Write the data to a file
			# Each of these files is formatted as follows:
			# <n bytes label> <m bytes pixel> <n bytes label> <m bytes pixel> <n bytes label><m bytes pixel>
			# when the number of class is n and the size of the one input sample is m

			collection.tofile(os.path.join(path, fname+".bin"))
			print('Finish writing data from %s'%(fname+".bin"))
		log_f.write('{}: {}\n'.format(pf, total))


def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    y = np.asarray(y, dtype='int32')
    # high dimensional array warning
    if len(y.shape) > 2:
        warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
    # flatten high dimensional array
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y

    
def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	num_preprocess_threads = 4
	if shuffle:
		images, labels = tf.train.shuffle_batch(
			[image,label], 
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + num_preprocess_threads * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, labels = tf.train.batch(
			[image,label], 
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + num_preprocess_threads * batch_size)

	#tf.summary.image('images', images)

	return images, tf.reshape(labels,[batch_size,-1])


class ReadData:
	def __init__(self, status, shape, set_id, subject_range):
		self.status = status
		self.image_shape = shape 
		self.set_id = set_id
		self.range = subject_range


	def read_from_files(self):
		
		data_dir = os.path.join(FLAGS.data_dir, self.set_id)
		filenames = os.listdir(data_dir)
		subject_index = TRAIN[self.range]
		#subject_index = TRAIN
		sel = []
		for fname in filenames:
			if fname[0:3] in subject_index:
				sel.append(fname)
		random.shuffle(sel)
		filenames = [os.path.join(data_dir, '{}'.format(i))
									for i in sel]
		

		# Create a queue that produce the filenmaest to read
		filename_queue = tf.train.string_input_producer(filenames)
		label, image = self.read_data(filename_queue)
		image = tf.cast(image, tf.float32)
		label = tf.cast(label, tf.int32)
		## Image processing and augumentation
		# When using 3D CNN
		# Image is a 4D tensor: [depth, height, width, channel]
		# We want to remove the channel dimension as the image is grayscale
		#image = tf.squeeze(image, [-1])

		# Convert from [depth, height, width] to [height, width, depth]
		#image = tf.transpose(image, [1, 2, 0])
		#image = tf.image.per_image_standardization(image) 

		# Normalized the image
		image = image/tf.reduce_max(image)
		image = image - tf.reduce_mean(image)


		# TODO: Find the suitable data augumentation methods
		#if self.status=='train':
		#	image = self.image_augumentation(image)   

		# Convert from [height, width, depth] to [depth, height, width]
		#image = tf.transpose(image, [2, 0, 1])
		#image = tf.expand_dims(image,-1)
		
		# Set the shapes of tensors.
		image.set_shape(self.image_shape)
		label.set_shape([FLAGS.label_bytes])

		# Ensure that the random shuffling has good mixing properties.
		
		min_queue_examples = int(FLAGS.num_train_images * FLAGS.min_fraction_of_examples_in_queue)

		print('Filling {} queue. This will take a few minutes.'.format(self.status))

		# Generate a batch of images and labels by building up a queue of examples.
		if self.status=='train':
			return _generate_image_and_label_batch(image, label, min_queue_examples, 
											batch_size=FLAGS.train_batch_size, shuffle=True)
		else:
			return _generate_image_and_label_batch(image, label, min_queue_examples, 
											batch_size=FLAGS.train_batch_size, shuffle=False)



	def read_data(self, filename_queue):
		image_bytes = 1
		for elem in self.image_shape:
			image_bytes *= elem

		label_bytes = FLAGS.label_bytes

		# Every record consists of a label followed by the image, with a
			# fixed number of bytes for each.
		record_bytes = label_bytes + image_bytes

		# Read a record, getting filenames from the filename_queue.
		reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
		key, value = reader.read(filename_queue)

		# Convert from a string to a vector of uint8 that is record_bytes long.
		record_bytes = tf.decode_raw(value, tf.uint8)

		# The first bytes represents the label
		label = tf.cast(
			tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

		image = tf.reshape(
			tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]), 
			self.image_shape)
		# Convert from [depth, height, width, channel] to [height, width, depth]

		return label, image


	def image_augumentation(self, image):
		
		#image = tf.image.random_flip_left_right(image)
		#image = tf.random_crop(image, [FLAGS.crop_size, FLAGS.crop_size, 1])

		#image = tf.image.resize_images(image, self.image_shape[0:2])
		image = tf.random_crop(image, [FLAGS.crop_size, FLAGS.crop_size,1])
		image = tf.image.resize_images(image, self.image_shape[0:2])

		#image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

		#angle = random.randint(-30,30)
		#image = tf.contrib.image.rotate(image,angle)
		# rotation
		return image



if __name__ == '__main__':
	seq_length = 10
	stride_frame = 5
	stride_seq = 7
	uniqname = 'noglasses_glasses_3D'

	set_id = '{}_{}_{}_{}'.format(uniqname,seq_length, stride_frame, stride_seq)
	get_data(set_id, seq_length, stride_frame, stride_seq, subject_index=TRAIN, scenarios=['noglasses', 'night_noglasses', 'glasses', 'night_glasses'])


