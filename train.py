import tensorflow as tf
import qrcodes
from qrcodes import IMAGE_SIZE

NUM_OUTPUTS = len(qrcodes.CHARACTER_SET)


# Helper functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name="weights")
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name="bias")
    return tf.Variable(initial)


# Validation set
NUM_TEST_IMAGES = 5000
print("Creating {} random test images ... ".format(NUM_TEST_IMAGES), end="", flush=True)
test_images, test_labels = qrcodes.getRandomBatch(size=NUM_TEST_IMAGES)
print("done")


# Inputs
with tf.name_scope("input"):
    x_image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="x_image")
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_OUTPUTS], name="y_")
    tf.summary.image('x_image', x_image, max_outputs=3)

with tf.name_scope("dropout_input"):
    keep_prob = tf.placeholder(tf.float32, name="keep_probability")

# Fully connected layer
NUM_FULLY_CONNECTED_1 = 128

with tf.name_scope("fc_1"):
    W_fc1 = weight_variable([IMAGE_SIZE * IMAGE_SIZE, NUM_FULLY_CONNECTED_1])
    b_fc1 = bias_variable([NUM_FULLY_CONNECTED_1])

    image_flat = tf.reshape(x_image, [-1, IMAGE_SIZE * IMAGE_SIZE])
    h_fc1 = tf.nn.relu(tf.matmul(image_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer two
NUM_FULLY_CONNECTED_2 = 64

with tf.name_scope("fc_2"):
    W_fc2 = weight_variable([NUM_FULLY_CONNECTED_1, NUM_FULLY_CONNECTED_2])
    b_fc2 = bias_variable([NUM_FULLY_CONNECTED_2])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Readout
with tf.name_scope("readout"):
    W_fcR = weight_variable([NUM_FULLY_CONNECTED_2, NUM_OUTPUTS])
    b_fcR = bias_variable([NUM_OUTPUTS])

    y_readout = tf.matmul(h_fc2_drop, W_fcR) + b_fcR

# Loss function
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_readout, labels=y_))
    tf.summary.scalar("cross entropy", cross_entropy)

# Training
with tf.name_scope("train_step"):
    train_step = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cross_entropy)

# Evaluation
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y_readout, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # Summaries
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter("/tmp/qrnet-log/train", sess.graph, flush_secs=5)
    test_writer = tf.summary.FileWriter("/tmp/qrnet-log/test", flush_secs=5)

    tf.global_variables_initializer().run()

    # saver = tf.train.Saver()

    # print("Trying to load model from checkpoint ... ", end="")
    # try:
    #     saver.restore(sess, "qrnet.chk")
    #     print("success")
    # except:
    #     print("failed")
    #     pass

    BATCH_SIZE = 200
    MAX_STEPS = 100000

    KEEP_PROBABILITY = 1.0

    for i in range(MAX_STEPS):
        batch_images, batch_labels = qrcodes.getRandomBatch(size=BATCH_SIZE)

        if i % 100 == 0:
            # Record test set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict={
                                    x_image: test_images,
                                    y_: test_labels,
                                    keep_prob: 1.0})
            print("Test set accuracy at step {:06}: {:.05}".format(i, acc), flush=True)
            test_writer.add_summary(summary, i)

            # Save model weights
            # saver.save(sess, "qrnet.chk")
        else:
            if i % 100 == 99:
                # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict={
                                      x_image: batch_images,
                                      y_: batch_labels,
                                      keep_prob: KEEP_PROBABILITY},
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
                                      keep_prob: KEEP_PROBABILITY})
                if i % 10 == 0:
                    train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
