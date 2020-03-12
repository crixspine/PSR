import gym
import numpy as np
import autoencoder.VanillaAutoEnc

# game: gym env, e.g. "MsPacman-ram-v0
# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: (Default = None)
# epochs: int
from autoencoder import VanillaAutoEnc
from keras.models import load_model

def getNumObservations(gameName):
    env = gym.make(gameName)
    observation_space = str(env.observation_space)
    return int(observation_space.split('(')[1].split(',')[0])

def getNumActions(gameName):
    env = gym.make(gameName)
    action_space = str(env.action_space)
    return int(action_space.split('(')[1].split(')')[0])

def trainInEnv(gameName, epochs):
    env = gym.make("MsPacman-ram-v0")
    size = getNumObservations(gameName)
    actions = getNumActions(gameName)
    print("Action Space: " + str(actions) + ", Observation Space: " + str(size))
    print(env.observation_space)

    for t in range(epochs):
        observation = env.reset()
        done = False
        obs_epoch = []
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation)
            obs_step = []
            for val in observation:
                obs_step.append(val)
            print(obs_step)
            obs_epoch.append(obs_step)
            # save observation

            # load model

            # if not os.path.exists("PSR/observations" + "\\Epoch " + str(epoch)):
            #     os.makedirs("PSR/observations" + "\\Epoch " + str(epoch))

            # pass observation to autoencoder
            # obs = VanillaAutoEnc.encode(observation,size)
            # update/save model


            # print(action)
            #save encoded into PSR

            if done:
                print("Finished training epoch " + str(t+1))
                print("observation: ", observation)
                print("reward: ", reward)
                print("info: ", info)
                print("done: ", done)
                print(obs_epoch)
                encoded_obs = VanillaAutoEnc.encode(obs_epoch, size)

                break
        env.close()

if __name__  == "__main__":
    train(game = "MsPacman-ram-v0", epochs = 10)