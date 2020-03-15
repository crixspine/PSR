import gym
import numpy as np
import autoencoder.VanillaAutoEnc

# game: gym env, e.g. "MsPacman-ram-v0
# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: (Default = None)
# epochs: int
from autoencoder import VanillaAutoEnc, DeepAutoEnc
from autoencoder import SimpleAutoEnc
from keras.models import load_model

def getNumObservations(gameName):
    env = gym.make(gameName)
    observation_space = str(env.observation_space)
    return int(observation_space.split('(')[1].split(',')[0])

def getNumActions(gameName):
    env = gym.make(gameName)
    action_space = str(env.action_space)
    return int(action_space.split('(')[1].split(')')[0])

def trainInEnv(gameName, iterNo, autoencoder):
    env = gym.make(gameName)
    size = getNumObservations(gameName)
    actions = getNumActions(gameName)
    if (iterNo == 0):
        print("Action Space: " + str(actions) + ", Observation Space: " + str(size))
    observation = env.reset()
    done = False
    obs_epoch = []
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        obs_step = []
        for val in observation:
            obs_step.append(val)
        obs_epoch.append(obs_step)
        if done:
            print("Finished training iteration " + str(iterNo+1))
            print("Observations from Gym for iteration " + str(iterNo+1) + ":")
            print(obs_epoch)
            if (autoencoder == 'simple'):
                if (iterNo == 0):
                    # train model only on the first epoch
                    encoded_obs = SimpleAutoEnc.trainModel(obs_epoch, size)
                else:
                    # load trained model from first epoch
                    encoded_obs = SimpleAutoEnc.encodeFromModel(obs_epoch)
            if (autoencoder == 'deep'):
                if (iterNo == 0):
                    # train model only on the first epoch
                    encoded_obs = DeepAutoEnc.trainModel(obs_epoch, size)
                else:
                    # load trained model from first epoch
                    encoded_obs = DeepAutoEnc.encodeFromModel(obs_epoch)
            # save encoded observations as integers in order to save as scalable PSR models
            # this incurs an additional loss of information after encoding (which already causes loss of information!)
            # might not need to convert if saving in small dimensions (code size in autoencoder as 1-3)
            # encoded_obs_int = encoded_obs.astype(int)
            # print(encoded_obs_int)
    env.close()
    return encoded_obs
