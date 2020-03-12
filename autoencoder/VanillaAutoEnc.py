import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

def calculateMaxTestId():
    return 10000  #return autoencoder maxdims

def saveModel():
    return
def loadModel():
    return

def encode(observation, size):
    tf.disable_v2_behavior()
    no_epochs = 1000
    lr = 0.2
    batch_size = 32

    n_in = size
    n_hidden = 32

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
        for e in range(no_epochs):
            np.random.shuffle(idx)
            X = np.array(X)[idx]

            cost_ = []
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X), batch_size)):
                _, cost__ = sess.run([train_op, mse], feed_dict={x: X[start:end]})
                cost_.append(cost__)

            cost.append(np.mean(cost_))

            if e % 100 == 0:
                print('epoch %d: cost %g' % (e, cost[e]))

        w = sess.run(W)
        h_, y_ = sess.run([h, y], {x: X})
    print(y_)

def decode(observations):

    return

