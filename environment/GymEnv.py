import gym
from autoencoder import DeepAutoEnc, SimpleAutoEnc
from bin import Parameter
from bin.Util import merge
from numpy.random import randint
from bin.MultiProcessSimulation import EvaluateMultiProcess, SimulateTrainDataMultiProcess, SimulateRunsOnCPSR

def getNumObservations(gameName):
    env = gym.make(gameName)
    observation_space = str(env.observation_space)
    return int(observation_space.split('(')[1].split(',')[0])

def getNumActions(gameName):
    env = gym.make(gameName)
    action_space = str(env.action_space)
    return int(action_space.split('(')[1].split(')')[0])

def SimulateTestingRun(self, runs, epoch, pool, psrModel, name, rewardDict, ns):
    args = []
    for i in range(Parameter.threadPoolSize):
        args.append([int(runs / Parameter.threadPoolSize), psrModel.ReturnEmptyObject(name=name), self.getNumActions(),
                     self.Clone(), epoch, randint(low=0, high=1000000000), rewardDict, ns])
    EvalDatas = pool.map(func=EvaluateMultiProcess, iterable=args)
    output = []
    for data in EvalDatas:
        output = output + data
    return output

def SimulateTrainData(self, runs, isRandom, psrModel, trainData, epoch, pool, RunOnVirtualEnvironment, name, rewardDict, ns):
    if not RunOnVirtualEnvironment:
        print("Simulating an agent on Real environment!")
        args = []
        for i in range(Parameter.threadPoolSize):
            args.append(
                [self.Clone(), psrModel.ReturnEmptyObject(name=name), int(runs / Parameter.threadPoolSize), isRandom,
                 self.getNumActions(), epoch, randint(low=0, high=1000000000), rewardDict, ns])
        TrainDataList = pool.map(func=SimulateTrainDataMultiProcess, iterable=args)
    else:
        print("Simulating an agent on CPSR environment!")
        args = []
        for i in range(Parameter.threadPoolSize):
            args.append(
                [psrModel.ReturnEmptyObject(name=name), int(runs / Parameter.threadPoolSize), self.getNumActions(),
                 epoch, randint(low=0, high=1000000000), rewardDict, ns])
        TrainDataList = pool.map(func=SimulateRunsOnCPSR, iterable=args)
    for TrainData in TrainDataList:
        trainData = merge(TrainData1=TrainData, OuputData=trainData)
    return trainData

# gameName: gym env, e.g. "MsPacman-ram-v0
# iterNo: training iteration no.
# autoencoder: 'simple' or 'deep'
def trainInEnv(gameName, iterNo, autoencoder):
    env = gym.make(gameName)
    size = getNumObservations(gameName)
    actions = getNumActions(gameName)
    if (iterNo == 0):
        print("Action Space: " + str(actions) + ", Observation Space: " + str(size))
    observation = env.reset()
    done = False
    obs_epoch = []
    actions_epoch = []
    rewards_epoch = []
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        obs_step = []
        for val in observation:
            obs_step.append(val)
        obs_epoch.append(obs_step)
        actions_epoch.append(action)
        rewards_epoch.append(reward)
        if done:
            print("Finished training iteration " + str(iterNo+1))
            print("Observations from Gym for iteration " + str(iterNo+1) + ":")
            print("Observations for epoch " + str(iterNo) + ":")
            print(obs_epoch)
            print("Actions for epoch " + str(iterNo) + ":")
            print(actions_epoch)
            print("Rewards for epoch " + str(iterNo) + ":")
            print(rewards_epoch)
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
            # save encoded observations as integers
            # this is needed as the encoded states are stored as the observation id in the dict
            # maintain the encoder precision by multiplying by a factor before converting to int
            encoded_obs_int = (encoded_obs * 100000).astype(int)
            print("Encoded observations for epoch " + str(iterNo) + ":")
            print(encoded_obs_int)
    env.close()
    return actions_epoch, encoded_obs_int, rewards_epoch
