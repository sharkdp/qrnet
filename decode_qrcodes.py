import glob
import os.path

import tensorflow as tf

import PIL.Image as Image
import numpy as np

IMAGE_SIZE = 21


# Helper functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name="weights")
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="bias")
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

        images[i, :, :, 0] = data
        num = int(os.path.splitext(os.path.basename(filename))[0])
        labels[i, :] = num_to_matrix(num)

        i += 1

    print("done", flush=True)

    return images, labels

train_images, train_labels = importData("train")
test_images, test_labels = importData("test")

# Inputs
with tf.name_scope("input"):
    x_image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_image")
    tf.image_summary('input_image', x_image, max_images=10)
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

# First convolutional layer
NUM_KERNELS_1 = 32
KERNEL_SIZE_1 = 5

with tf.name_scope("conv_1"):
    W_conv1 = weight_variable([KERNEL_SIZE_1, KERNEL_SIZE_1, 1, NUM_KERNELS_1])
    b_conv1 = bias_variable([NUM_KERNELS_1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
NUM_KERNELS_2 = 64
KERNEL_SIZE_2 = 5

with tf.name_scope("conv_2"):
    W_conv2 = weight_variable([KERNEL_SIZE_2, KERNEL_SIZE_2, NUM_KERNELS_1, NUM_KERNELS_2])
    b_conv2 = bias_variable([NUM_KERNELS_2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
NUM_FULLY_CONNECTED_1 = 1024

with tf.name_scope("fc_1"):
    W_fc1 = weight_variable([KERNEL_SIZE_2 * KERNEL_SIZE_2 * NUM_KERNELS_2, NUM_FULLY_CONNECTED_1])
    b_fc1 = bias_variable([NUM_FULLY_CONNECTED_1])

    h_pool2_flat = tf.reshape(h_pool2, [-1, KERNEL_SIZE_2 * KERNEL_SIZE_2 * NUM_KERNELS_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name="keep_probability")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout
with tf.name_scope("readout"):
    W_fcR = weight_variable([NUM_FULLY_CONNECTED_1, 10])
    b_fcR = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fcR) + b_fcR

# Loss function
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    tf.scalar_summary("cross entropy", cross_entropy)

# Training
with tf.name_scope("train_step"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluation
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary("accuracy", accuracy)

with tf.Session() as sess:
    # Summaries
    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter('log/train', sess.graph, flush_secs=5)
    test_writer = tf.train.SummaryWriter('log/test', flush_secs=5)

    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    # print("Loading model from checkpoint")
    # saver.restore(sess, "test.chk")

    BATCH_SIZE = 20
    MAX_STEPS = 40000

    for i in range(MAX_STEPS):
        b = (BATCH_SIZE * i) % 8000
        batch_images = train_images[b:(b + BATCH_SIZE)]
        batch_labels = train_labels[b:(b + BATCH_SIZE)]

        if i % 100 == 0:
            # Record test set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict={
                                    x_image: test_images,
                                    y_: test_labels,
                                    keep_prob: 1.0})
            print("Test accuracy at step {:06}: {:.03}".format(i, acc), flush=True)
            test_writer.add_summary(summary, i)
        else:
            if i % 100 == 99:
                # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict={
                                      x_image: batch_images,
                                      y_: batch_labels,
                                      keep_prob: 0.5},
                                      options=run_options,
                                      run_metadata=run_metadata
                                      )
                train_writer.add_run_metadata(run_metadata, "step_{:06}".format(i))
                train_writer.add_summary(summary, i)
            else:
                # Training
                summary, _ = sess.run([merged, train_step], feed_dict={
                                      x_image: batch_images,
                                      y_: batch_labels,
                                      keep_prob: 0.5})
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()

    saver.save(sess, "qrnet.chk")
