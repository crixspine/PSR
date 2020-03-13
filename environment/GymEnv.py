import gym
import numpy as np
import autoencoder.VanillaAutoEnc

# game: gym env, e.g. "MsPacman-ram-v0
# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: (Default = None)
# epochs: int
from autoencoder import VanillaAutoEnc
from autoencoder import DeepAutoEnc
from keras.models import load_model

def getNumObservations(gameName):
    env = gym.make(gameName)
    observation_space = str(env.observation_space)
    return int(observation_space.split('(')[1].split(',')[0])

def getNumActions(gameName):
    env = gym.make(gameName)
    action_space = str(env.action_space)
    return int(action_space.split('(')[1].split(')')[0])

def trainInEnv(gameName, iterNo):
    env = gym.make("MsPacman-ram-v0")
    size = getNumObservations(gameName)
    actions = getNumActions(gameName)
    print("Action Space: " + str(actions) + ", Observation Space: " + str(size))
    print(env.observation_space)

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
            print("Finished training epoch " + str(iterNo))
            print(obs_epoch)
            if (iterNo == 0):
                # train model only on the first epoch
                encoded_obs = VanillaAutoEnc.trainModel(obs_epoch, size)
            else:
                encoded_obs = VanillaAutoEnc.encodeFromModel(obs_epoch, size)
            break

    env.close()
    return encoded_obs

if __name__  == "__main__":
    train(game = "MsPacman-ram-v0", epochs = 10)