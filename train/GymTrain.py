from autoencoder import VanillaAutoEnc
from environment import GymEnv
from model.PSRmodel import CompressedPSR
from bin.TrainingData import TrainingData
from bin.Agent import Agent
from bin import Parameter
from multiprocessing import Pool, Manager, Lock
from bin.Util import ConvertToTrainSet, writerMemoryintodisk
import os
import numpy as np
from bin.MultiProcessSimulation import init

import environment.GymEnv

# model: CPSR, TPSR (default = CPSR)
# policy: fitted-Q, DRL
# encoder: (Default = None)
# epochs: int


def WriteEvalUateDataForGym(EvalData, epoch):
    if not os.path.exists("../observations" + "\\Epoch " + str(epoch)):
        os.makedirs("../observations" + "\\Epoch " + str(epoch))
    with open(file="../observations" + "\\Epoch " + str(epoch) + "\\summary", mode='w') as f:
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
    # TrainingData.LoadData(TrainData=trainData, file="../observations/RandomSampling.txt", rewardDict=rewardDict)
    TrainingData.LoadData(TrainData=trainData, file="PSR/observations/RandomSampling.txt", rewardDict=rewardDict)
    for i in range(epoch):
        trainData.newDataBatch()
        TrainingData.LoadData(TrainData=trainData, file="epilsonGreedySampling" + str(i) + ".txt",
                              rewardDict=rewardDict)

import environment.GymEnv
import time
from bin.Util import ConvertLastBatchToTrainSet, readMemoryfromdisk, copyRewardDict
import autoencoder.VanillaAutoEnc

def train(gameName, epochs):
    #TODO: write all inputs to params file
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    ns.rewardCount = 0
    # file = "../train/setting/Gym.json"
    file = "PSR/train/setting/Gym.json"
    Parameter.readfile(file=file)
    # RandomSamplingForPSR = True
    # isbuiltPSR = True
    Parameter.maxTestID = VanillaAutoEnc.calculateMaxTestId()
    trainData = TrainingData()
    iterNo = 0
    print("No. of iterations to run: " + str(epochs))
    agent = Agent(PnumActions=GymEnv.getNumActions(gameName), epsilon=Parameter.epsilon,
                  inputDim=(Parameter.svdDim,), algorithm=Parameter.algorithm, Parrallel=True)
    print("Learning algorithm/Policy: " + Parameter.algorithm)

    # rdict = readMemoryfromdisk(file="../bin/rewardDict.txt")
    rdict = readMemoryfromdisk(file="PSR/bin/rewardDict.txt")
    copyRewardDict(rewardDict=rewardDict, rewardDict1=rdict)
    psrModel = CompressedPSR("Gym")
    psrPool = Pool(Parameter.threadPoolSize, initializer=init, initargs=(Parameter.maxTestID, file, Lock(),))
    print("Finishing Preparation!")
    loadCheckPoint(trainData=trainData, epoch=iterNo, psrModel=psrModel, rewardDict=rewardDict)
    trainData = trainData.MergeAllBatchData()
    trainSet = None
    print("Game environment in gym: " + gameName)
    while iterNo < epochs:
    #     print("Starting Iteration: " + str(iterNo + 1))
    #     if RandomSamplingForPSR:
    #         trainData.newDataBatch()
    #
    #         #edit
    #         game.SimulateTrainData(runs=Parameter.runsForCPSR, isRandom=True, psrModel=psrModel,
    #                                trainData=trainData, epoch=iterNo - 1, pool=psrPool,
    #                                RunOnVirtualEnvironment=False, name=game.getGameName(), rewardDict=rewardDict,
    #                                ns=ns)
    #
    #         psrModel.validActObset = trainData.validActOb
    #         WriteEvalUateDataForPacMan(EvalData=trainData.data[trainData.getBatch()], epoch=-1)
    #         # trainData.WriteData(file="../observations" + "\\RandomSampling" + str(iterNo) + ".txt")
    #         trainData.WriteData(file="PSR/observations" + "\\RandomSampling" + str(iterNo) + ".txt")
    #         RandomSamplingForPSR = False
    #     if isbuiltPSR:
        states = GymEnv.trainInEnv(gameName, 1)
        print(states)
        psrModel.build(data=trainData, aos=trainData.validActOb, pool=psrPool, rewardDict=rewardDict)
        psrModel.saveModel(epoch=iterNo)
        # writerMemoryintodisk(file="../bin/rewardDict.txt", data=rewardDict.copy())
        writerMemoryintodisk(file="PSR/bin/rewardDict.txt", data=rewardDict.copy())
        print("Convert sampling data into training forms")
        if trainSet is None:
            trainSet = ConvertToTrainSet(data=trainData, RewardDict=rewardDict,
                                         pool=psrPool, epoch=iterNo, name=game.getGameName(), psrModel=psrModel)
        else:
            trainSet = trainSet + ConvertLastBatchToTrainSet(data=trainData, RewardDict=rewardDict,
                                                             pool=psrPool, epoch=iterNo, name=game.getGameName(),
                                                             psrModel=psrModel)
        print("Starting training")
        tick1 = time.time()
        print("Iteration: %d/%d"%(iterNo+1, epochs))
        agent.Train_And_Update(data=trainSet, epoch=iterNo, pool=psrPool)
        tick2 = time.time()
        print("The time spent on training:" + str(tick2 - tick1))
        agent.SaveWeight(epoch=iterNo)
        print("Evaluating the agent")
        tick3 = time.time()
        EvalData = game.SimulateTestingRun(runs=Parameter.testingRuns, epoch=iterNo, pool=psrPool,  #edit
                                           psrModel=psrModel, name=game.getGameName(), rewardDict=rewardDict, ns=ns) #edit
        tick4 = time.time()
        print("The time spent on Evaluate:" + str(tick4 - tick3))
        trainData.newDataBatch()

        #edit
        game.SimulateTrainData(runs=Parameter.runsForLearning, psrModel=psrModel, trainData=trainData,
                               isRandom=False, epoch=iterNo, pool=psrPool,
                               RunOnVirtualEnvironment=Parameter.trainingOnVirtualEnvironment,
                               name=game.getGameName(), rewardDict=rewardDict, ns=ns)

        # trainData.WriteData(file="../observations/epsilonGreedySampling" + str(iterNo) + ".txt")
        trainData.WriteData(file="PSR/observations/epsilonGreedySampling" + str(iterNo) + ".txt")

        WriteEvalUateDataForPacMan(EvalData=EvalData, epoch=iterNo)
        iterNo = iterNo + 1
