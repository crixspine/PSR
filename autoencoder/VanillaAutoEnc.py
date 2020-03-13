import tensorflow.compat.v1 as tf
import numpy as np


def calculateMaxTestId():
    return 10000  #return autoencoder maxdims

def encodeFromModel(observation, size):
    print("Loading trained AutoEncoder model")
    tf.disable_v2_behavior()
    no_epochs = 1000
    lr = 0.2
    batch_size = 32
    n_in = size
    n_hidden = 32

    encoded = []
    X = observation
    x = tf.placeholder(tf.float32, [None, n_in])

    W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1 / np.sqrt(n_in)))
    b = tf.Variable(tf.zeros([n_hidden]))
    b_prime = tf.Variable(tf.zeros([n_in]))
    W_prime = tf.transpose(W)

    h = tf.sigmoid(tf.matmul(x, W) + b)
    y = tf.sigmoid(tf.matmul(h, W_prime) + b_prime)

    imported_graph = tf.train.import_meta_graph('AutoEncoderModel-999.meta')

    with tf.Session() as sess:
        # restore the saved model
        # saver.restore(sess, './AutoEncoderModel')
        imported_graph.restore(sess, './AutoEncoderModel')
        print("AutoEncoder model loaded.")
        h_, y_ = sess.run([h, y], {x: X})
        for val in y_:
            encoded.append(val)
    return encoded

def trainModel(observation, size):
    print("Training AutoEncoder model")
    tf.disable_v2_behavior()
    no_epochs = 1000
    lr = 0.2
    batch_size = 32
    n_in = size
    n_hidden = 32

    encoded = []
    X = observation
    x = tf.placeholder(tf.float32, [None, n_in])

    W = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=1 / np.sqrt(n_in)))
    b = tf.Variable(tf.zeros([n_hidden]))
    b_prime = tf.Variable(tf.zeros([n_in]))
    W_prime = tf.transpose(W)

    h = tf.sigmoid(tf.matmul(x, W) + b)
    y = tf.sigmoid(tf.matmul(h, W_prime) + b_prime)

    mse = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=1))
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(mse)

    idx = np.arange(len(X))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cost = []
        print("Evaluating cost of training model over " + str(no_epochs) + " epochs")
        for e in range(no_epochs):
            np.random.shuffle(idx)
            X = np.array(X)[idx]
            cost_ = []
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
                _, cost__ = sess.run([train_op, mse], feed_dict={x: X[start:end]})
                cost_.append(cost__)
            cost.append(np.mean(cost_))
            if e % 100 == 0:
                print('Epoch %d: Cost %g' % (e, cost[e]))

        w = sess.run(W)
        h_, y_ = sess.run([h, y], {x: X})
        for val in y_:
            encoded.append(val)
        saver = tf.train.Saver()
        saver.save(sess, './AutoEncoderModel', global_step=e)

        return encoded

def decode(observations):

    return

