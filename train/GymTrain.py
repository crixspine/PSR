from autoencoder import SimpleAutoEnc, DeepAutoEnc
from environment import GymEnv
from model.PSRmodel import CompressedPSR
from bin.TrainingData import TrainingData
from bin.Agent import Agent
from bin import Parameter
from multiprocessing import Pool, Manager, Lock
from bin.Util import ConvertToTrainSet, writeMemoryintodisk
import os
import numpy as np
from bin.MultiProcessSimulation import init

def WriteEvalUateDataForGym(EvalData, epoch):
    dir = os.path.abspath(os.getcwd())
    if not os.path.exists(dir + "/observations/Epoch " + str(epoch)):
        os.makedirs(dir + "/observations/Epoch " + str(epoch))
    with open(file=dir + "/observations/Epoch " + str(epoch) + "/summary", mode='w') as f:
        TotalRewards = []
        lenActions = []
        count_wins = 0
        for Episode in EvalData:
            EpisodeRewards = 0
            EpisodeLength = 0
            for ActOb in Episode:
                a = ActOb[0]
                if a == -1:
                    break
                r = ActOb[2]
                EpisodeRewards = EpisodeRewards + r
                EpisodeLength = EpisodeLength + 1
                if r >= 100:
                    count_wins = count_wins + 1
            lenActions.append(EpisodeLength)
            TotalRewards.append(EpisodeRewards)
        averageValue = np.mean(a=TotalRewards, axis=0)
        variance = np.var(TotalRewards)
        std = np.std(TotalRewards)
        # how many actions the agent takes before making final decision
        f.write("Average Value For Each Episode: " + str(averageValue) + '\n')
        f.write("The Variance of EpisodeReward: " + str(variance) + '\n')
        f.write("The Standard Variance of EpisodeReward: " + str(std) + '\n')
        f.write("The average length of a game is: " + str(np.mean(a=lenActions, axis=-1)) + "\n")
        if count_wins != 0:
            f.write("The wining Probability:" + str(len(TotalRewards) / count_wins))
        else:
            f.write("The wining Probability:" + str(-1))

def loadCheckPoint(trainData, psrModel, epoch, rewardDict):
    trainData.newDataBatch()
    TrainingData.LoadData(TrainData=trainData, file="PSR/RandomSampling.txt", rewardDict=rewardDict)
    for i in range(epoch):
        trainData.newDataBatch()
        TrainingData.LoadData(TrainData=trainData, file="epsilonGreedySampling" + str(i) + ".txt",
                              rewardDict=rewardDict)

import time
from bin.Util import ConvertLastBatchToTrainSet, readMemoryfromdisk, copyRewardDict

# gameName = game from Gym env
# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: 'simple' or 'deep'
# epochs: int
def train(gameName, epochs, autoencoder):
    dir = os.path.abspath(os.getcwd())
    print("Current Working Directory: " + dir)
    #TODO: write all inputs to params file
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    if not os.path.exists(dir + "/tmp"):
        os.makedirs(dir + "/tmp")
    ns.rewardCount = 0
    file = "PSR/train/setting/Gym.json"
    Parameter.readfile(file=file)
    # RandomSamplingForPSR = True
    # isbuiltPSR = True
    if (autoencoder == 'simple'):
        Parameter.maxTestID = SimpleAutoEnc.calculateMaxTestId()
    if (autoencoder == 'deep'):
        Parameter.maxTestID = DeepAutoEnc.calculateMaxTestId()
    trainData = TrainingData()
    iterNo = 0
    print("No. of iterations to run: " + str(epochs))
    agent = Agent(PnumActions=GymEnv.getNumActions(gameName), epsilon=Parameter.epsilon,
                  inputDim=(Parameter.svdDim,), algorithm=Parameter.algorithm, Parrallel=True)
    print("Learning algorithm/Policy: " + Parameter.algorithm)

    rdict = readMemoryfromdisk(file="PSR/rewardDict.txt")
    copyRewardDict(rewardDict=rewardDict, rewardDict1=rdict)
    psrModel = CompressedPSR("Gym")
    psrPool = Pool(Parameter.threadPoolSize, initializer=init, initargs=(Parameter.maxTestID, file, Lock(),))
    print("Finishing Preparation!")
    loadCheckPoint(trainData=trainData, epoch=iterNo, psrModel=psrModel, rewardDict=rewardDict)
    trainData = trainData.MergeAllBatchData()
    trainSet = None
    print("Game environment in gym: " + gameName)
    while iterNo < epochs:
        print("Starting Iteration: " + str(iterNo + 1))
        states = GymEnv.trainInEnv(gameName, iterNo, autoencoder)
        print(states)
        # psrModel.build(data=trainData, aos=trainData.validActOb, pool=psrPool, rewardDict=rewardDict)
        # psrModel.saveModel(epoch=iterNo)
        # # writerMemoryintodisk(file="../bin/rewardDict.txt", data=rewardDict.copy())
        # writerMemoryintodisk(file="PSR/bin/rewardDict.txt", data=rewardDict.copy())
        # print("Convert sampling data into training forms")
        # if trainSet is None:
        #     trainSet = ConvertToTrainSet(data=trainData, RewardDict=rewardDict,
        #                                  pool=psrPool, epoch=iterNo, name=game.getGameName(), psrModel=psrModel)
        # else:
        #     trainSet = trainSet + ConvertLastBatchToTrainSet(data=trainData, RewardDict=rewardDict,
        #                                                      pool=psrPool, epoch=iterNo, name=game.getGameName(),
        #                                                      psrModel=psrModel)
        # print("Starting training of Agent")
        # tick1 = time.time()
        # print("Iteration: %d/%d"%(iterNo+1, epochs))
        # agent.Train_And_Update(data=trainSet, epoch=iterNo, pool=psrPool)
        # tick2 = time.time()
        # print("The time spent on training:" + str(tick2 - tick1))
        # agent.SaveWeight(epoch=iterNo)
        # print("Evaluating the agent")
        # tick3 = time.time()
        # EvalData = game.SimulateTestingRun(runs=Parameter.testingRuns, epoch=iterNo, pool=psrPool,  #edit
        #                                    psrModel=psrModel, name=game.getGameName(), rewardDict=rewardDict, ns=ns) #edit
        # tick4 = time.time()
        # print("The time spent on Evaluate:" + str(tick4 - tick3))
        # trainData.newDataBatch()
        #
        # #edit
        # game.SimulateTrainData(runs=Parameter.runsForLearning, psrModel=psrModel, trainData=trainData,
        #                        isRandom=False, epoch=iterNo, pool=psrPool,
        #                        RunOnVirtualEnvironment=Parameter.trainingOnVirtualEnvironment,
        #                        name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        #
        # # trainData.WriteData(file="../observations/epsilonGreedySampling" + str(iterNo) + ".txt")
        # trainData.WriteData(file="PSR/observations/epsilonGreedySampling" + str(iterNo) + ".txt")
        # WriteEvalUateDataForGym(EvalData=EvalData, epoch=iterNo)
        iterNo = iterNo + 1