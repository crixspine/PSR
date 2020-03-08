import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

def saveModel:
    return
def loadModel:
    return

def encode(observations):
    input_size = 128
    hidden_size = 32
    output_size = 8

    x = Input(shape=(input_size,))

    # Encoder
    h = Dense(hidden_size, activation='relu')(x)

    # Decoder
    r = Dense(output_size, activation='sigmoid')(h)

    autoencoder = Model(input=x, output=r)
    autoencoder.compile(optimizer='adam', loss='mse')

    return

def decode(observations):

    return

