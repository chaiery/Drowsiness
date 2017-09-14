import tensorflow as tf
import tflearn_dev as tflearn


##################################################################################################################

def _conv3d(x, depth_size, spatial_size, out, name):
    shape = [depth_size, spatial_size, spatial_size, x.shape[-1], out]
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[shape[-1]], initializer=tf.constant_initializer(0))
        out = tf.add(tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME'), b)
        activation = tf.nn.relu(out)
        return activation


def _conv2d(x, spatial_size, out, name):
    '''
    shape = [filter_size, filter_size, number of channel, number of filter]
    '''
    shape = [spatial_size, spatial_size, x.shape[-1], out]

    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[shape[3]], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        activation = tf.nn.relu(out)
        return activation


def _max_pool_2x2(x, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def _max_pool_3d_2x2(x, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                              strides=[1, 2, 2, 2, 1], padding='SAME')


def _fully_connected(x, num_neuro, name):
    '''
    x: [batch_size, spatial_size, spatial_size, channel]
    '''
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        W = tf.get_variable(name='W', shape=[dim,num_neuro], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[num_neuro], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.matmul(x, W), b)

        return out



##################################################################################################################


def baseline_3D(inputs, wd):
    # filter size: [filter_depth, filter_height, filter_width, in_channels, out_channels]

    network = _conv3d(inputs, 3, 3, 16, 'conv1')
    #network = _conv3d_rescale(network, 3, 3, 16, 'conv2')
    network = _max_pool_3d_2x2(network, 'pool_1')
    network = tflearn.batch_normalization(network)

    network = _conv3d(network, 3, 3, 32, 'conv3')
    #network = _conv3d_rescale(network, 3, 3, 32, 'conv4')
    network = _max_pool_3d_2x2(network, 'pool_2')
    network = tflearn.batch_normalization(network)

    #network = _conv3d(network, 3, 3, 32, 'conv5')
    network = _conv3d(network, 3, 3, 64, 'conv6')
    network = _max_pool_3d_2x2(network, 'pool_3')
    network = tflearn.batch_normalization(network)

    network = _conv3d(network, 3, 3, 128, 'conv7')
    network = _max_pool_3d_2x2(network, 'pool_3')
    network = tflearn.batch_normalization(network)


    network = _fully_connected(network, 50, 'fc_1')
    network = tf.nn.relu(network)
    network = tf.nn.dropout(network, 0.5, name='drop_1')

    network = _fully_connected(network, 2, 'fc_2')
    return network


def baseline_2D(inputs, wd):
    network = _conv2d(inputs, 3,32, 'conv_1')
    #network = _conv2d(network, 3,32, 'conv_2')
    network = _max_pool_2x2(network, 'pool_1')

    network = tflearn.batch_normalization(network, scope='bn1')

    network = _conv2d(network, 3,64, 'conv_3')
    #network = _conv2d(network, 3,64, 'conv_4')
    network = _max_pool_2x2(network, 'pool_1')
    network = tflearn.batch_normalization(network, scope='bn2')

    network = _conv2d(network, 3,128, 'conv_5')
    network = _max_pool_2x2(network, 'pool_2')

    network = _fully_connected(network, 100, 'fc_1')
    network = tf.nn.relu(network)
    network = tf.nn.dropout(network, 0.5, name='drop_1')

    network = _fully_connected(network, 2, 'fc_2')
    #network = tf.nn.softmax(network, name='softmax')
    return network