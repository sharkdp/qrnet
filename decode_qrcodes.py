import glob
import os.path

import tensorflow as tf

import PIL.Image as Image
import numpy as np

IMAGE_SIZE = 20


# Helper functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def digit_to_vector(d):
    vec = np.zeros((10,), dtype=np.float)
    vec[d] = 1.0
    return vec


def num_to_matrix(num):
    numStr = "{:04}".format(num)

    return digit_to_vector(int(numStr[0]))


def importData(folder):
    paths = glob.glob(os.path.join(folder, "*.png"))

    numImages = len(paths)

    print("Importing {} images from '{}' ... ".format(numImages, folder), end="", flush=True)

    images = np.zeros((numImages, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float)
    labels = np.zeros((numImages, 10), dtype=np.int)
    i = 0
    for filename in paths:
        data = np.asarray(Image.open(filename), dtype=np.float)
        # crop from 21 x 21 to 20 x 20
        data = data[0:IMAGE_SIZE, 0:IMAGE_SIZE]

        images[i, :, :, 0] = data
        num = int(os.path.splitext(os.path.basename(filename))[0])
        labels[i, :] = num_to_matrix(num)

        i += 1

    print("done", flush=True)

    return images, labels

train_images, train_labels = importData("train")
test_images, test_labels = importData("test")

x_image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_image")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

# Model

# First convolutional layer
NUM_KERNELS_1 = 32

W_conv1 = weight_variable([5, 5, 1, NUM_KERNELS_1])
b_conv1 = bias_variable([NUM_KERNELS_1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
NUM_KERNELS_2 = 64

W_conv2 = weight_variable([5, 5, NUM_KERNELS_1, NUM_KERNELS_2])
b_conv2 = bias_variable([NUM_KERNELS_2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
NUM_FULLY_CONNECTED = 1024

W_fc1 = weight_variable([5 * 5 * NUM_KERNELS_2, NUM_FULLY_CONNECTED])
b_fc1 = bias_variable([NUM_FULLY_CONNECTED])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * NUM_KERNELS_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout
W_fc2 = weight_variable([NUM_FULLY_CONNECTED, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# Training


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    # print("Loading model from checkpoint")
    # saver.restore(sess, "test.chk")

    test_accuracy = accuracy.eval(feed_dict={
                                  x_image: test_images,
                                  y_: test_labels,
                                  keep_prob: 1.0})

    print("testing accuracy {:.03}".format(test_accuracy), flush=True)

    batchSize = 20
    for i in range(4000):
        b = (batchSize * i) % 8000
        batch = train_images[b:(b + batchSize)]
        batch_labels = train_labels[b:(b + batchSize)]

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                                           x_image: batch,
                                           y_: batch_labels,
                                           keep_prob: 1.0})
            print("step {:05}, training accuracy {:.03}".format(i, train_accuracy))
            # test_accuracy = accuracy.eval(feed_dict={
            #                               x_image: test_images,
            #                               y_: test_labels,
            #                               keep_prob: 1.0})

            # print("{:05}\t{:.03}".format(i, test_accuracy), flush=True)

        train_step.run(feed_dict={x_image: batch, y_: batch_labels, keep_prob: 0.5})

    saver.save(sess, "test.chk")

    test_accuracy = accuracy.eval(feed_dict={
                                  x_image: test_images,
                                  y_: test_labels,
                                  keep_prob: 1.0})

    print("testing accuracy {:.03}".format(test_accuracy), flush=True)
