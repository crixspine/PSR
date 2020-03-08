import gym

# game: gym env, e.g. "MsPacman-ram-v0
# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: (Default = None)
# epochs: int

def train(game, epochs):
    env = gym.make("MsPacman-ram-v0")
    print(env.action_space)
    print(env.observation_space)

    for t in range(epochs):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            # save observation
            # pass observation to autoencoder
            action = env.action_space.sample()
            # print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Finished training epoch " + str(t+1))
                print("observation: ", observation)
                print("reward: ", reward)
                print("info: ", info)
                print("done: ", done)
                break
        env.close()

if __name__  == "__main__":
    train(game = "MsPacman-ram-v0", epochs = 10)