# keras implementation; simpler and more lightweight
import os
from keras.models import Model, load_model
from keras.layers import Dense, Input

def calculateMaxTestId():
    # return encoded code max dimensions/permutations
    # permutations of each single value ^ code_size
    # example observation: [   0.   545.79706   0.   0.   ]
    # example observation fot code_size=1: [   0.   471.27356   0.   0.   ]
    return (10**8)**1

def encodeFromModel(observation):
    dir = os.path.abspath(os.getcwd())
    print("Loading Encoder...")
    input = [observation]
    if os.path.exists(dir + "/AutoEncModel"):
        model = load_model(dir + '/AutoEncModel/EncoderNetwork', compile=False)
    else:
        Exception("AutoEncoder model not found!")
    return model.predict(input)

def trainModel(observation, input_size):
    dir = os.path.abspath(os.getcwd())
    input = [observation]
    code_size = 1
    input_obs = Input(shape=(input_size,))

    # representation of encoder and decoder networks
    encoded = Dense(code_size, activation='relu')(input_obs)
    decoded = Dense(input_size, activation='relu')(encoded)

    # implementing encoder and decoder model
    autoencoder = Model(input_obs, decoded)
    encoder = Model(input_obs, encoded)
    encoded_input = Input(shape=(code_size,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    print("Training Autoencoder Model...")
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(input, input,
                    epochs=100,
                    shuffle=True,
                    verbose=0)
    print("Evaluating Autoencoder Model...")
    autoencoder.evaluate(input, input,
                         verbose=1)
    print("Saving Autoencoder model")
    if not os.path.exists(dir + "/AutoEncModel"):
        os.makedirs(dir + "/AutoEncModel")
    encoder.save(dir + '/AutoEncModel/EncoderNetwork')
    decoder.save(dir + '/AutoEncModel/DecoderNetwork')
    print('Encoded Observations: ')
    return encoder.predict(input)

def decode(observation):
    dir = os.path.abspath(os.getcwd())
    print("Loading Decoder...")
    input = [observation]
    if os.path.exists(dir + "/AutoEncModel"):
        model = load_model(dir + '/AutoEncModel/DecoderNetwork')
    else:
        Exception("AutoEncoder model not found!")
    print('Decoded Observations: ')
    return model.predict(input)

