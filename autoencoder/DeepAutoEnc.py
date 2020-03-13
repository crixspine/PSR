import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1
from keras.optimizers import Adam

def calculateMaxTestId():
    return 10000  #return autoencoder maxdims

def saveModel():
    return
def loadModel():
    return

def encode(observation, size):
    input_size = size
    hidden_size = 32
    code_size = 8

    input_obs = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_obs)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_obs = Dense(input_size, activation='sigmoid')(hidden_2)

    autoencoder = Model(input_obs, output_obs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(observation, observation, epochs=5)

def decode(observations):

    return

