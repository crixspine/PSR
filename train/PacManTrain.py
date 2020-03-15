from model.PSRmodel import CompressedPSR
from bin.TrainingData import TrainingData
from bin.Agent import Agent
from bin import Parameter
from multiprocessing import Pool, Manager, Lock
from bin.Util import ConvertToTrainSet
import os
import numpy as np
from bin.MultiProcessSimulation import init

def WriteEvalUateDataForPacMan(EvalData, epoch):
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

def WriteEvalUateData(EvalData, Env, epoch):
    if not os.path.exists(dir + "/observations/Epoch " + str(epoch)):
        os.makedirs(dir + "/observations/Epoch " + str(epoch))
    with open(file=dir + "/observations/Epoch " + str(epoch) + "/summary", mode='w') as f:
        with open(file=dir + "/observations/Epoch " + str(epoch) + "/trajectory", mode='w') as f1:
            TotalRewards = []
            winTimes = 0
            failTime = 0
            lenActions = []
            for Episode in EvalData:
                EpisodeRewards = 0
                winTimesEpisode = 0
                failTimesEpisode = 0
                for ActOb in Episode:
                    if ActOb[0] == -1:
                        continue
                    a = Env.Actions[ActOb[0]]
                    o = Env.Observations[ActOb[1]]
                    r = Env.Rewards[ActOb[2]]
                    EpisodeRewards = EpisodeRewards + r
                    f1.write(a + " " + o + " " + str(r) + ",")
                    if Env.getGameName() == "Tiger95":
                        if r == 10:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100:
                            failTimesEpisode = failTimesEpisode + 1
                        else:
                            if r != -1:
                                Exception("reward" + str(r) + "are not seen")
                    elif Env.getGameName() == "Maze":
                        if r == 10.0:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100.0:
                            failTimesEpisode = failTimesEpisode + 1
                    elif Env.getGameName() == "StandTiger":
                        if r == 30:
                            winTimesEpisode = winTimesEpisode + 1
                        elif r == -100:
                            failTimesEpisode = failTimesEpisode + 1
                winTimes = winTimes + winTimesEpisode
                failTime = failTime + failTimesEpisode
                if winTimesEpisode + failTimesEpisode != 0:
                    lenActions.append(Parameter.lengthOfAction / (winTimesEpisode + failTimesEpisode))
                TotalRewards.append(EpisodeRewards)
                f1.write('\n')
            averageValue = np.mean(a=TotalRewards, axis=0)
            variance = np.var(TotalRewards)
            std = np.std(TotalRewards)
            if (winTimes + failTime) != 0:
                winProb = winTimes / (winTimes + failTime)
            else:
                winProb = 0
            # how many actions the agent takes before making final decision
            f.write("Average Value For Each Episode: " + str(averageValue) + '\n')
            f.write("The Variance of EpisodeReward: " + str(variance) + '\n')
            f.write("The Standard Variance of EpisodeReward: " + str(std) + '\n')
            f.write("The Winning Probability of the agent: " + str(winProb) + '\n')
            if len(lenActions) == 0:
                w = -1
            else:
                w = np.mean(a=lenActions, axis=-1)
            f.write("The steps of actions the agent takes before making final decision: " + str(w) + '\n')

def loadCheckPoint(trainData, psrModel, epoch, rewardDict):
    dir = os.path.abspath(os.getcwd())
    trainData.newDataBatch()
    TrainingData.LoadData(TrainData=trainData, file="PSR/RandomSampling.txt", rewardDict=rewardDict)
    for i in range(epoch):
        trainData.newDataBatch()
        TrainingData.LoadData(TrainData=trainData, file=dir + "/observations/epsilonGreedySampling" + str(i) + ".txt",
                              rewardDict=rewardDict)

import sys
from environment.PacMan import PacMan
import time
from bin.Util import ConvertLastBatchToTrainSet, readMemoryfromdisk, copyRewardDict

vars = sys.float_info.min

# model: CPSR, TPSR (default = CPSR)
# epochs: int
# policy: fitted-Q, DRL
def train(epochs, policy):
    dir = os.path.abspath(os.getcwd())
    print("Current Working Directory: " + dir)
    if not os.path.exists(dir + "/tmp"):
        os.makedirs(dir + "/tmp")
    manager = Manager()
    rewardDict = manager.dict()
    ns = manager.Namespace()
    ns.rewardCount = 0
    file = "PSR/train/setting/PacMan.json"
    if (policy == "fitted_Q"):
        Parameter.edit(file=file, param="algorithm", newval="fitted_Q")
    elif (policy == "DRL"):
        Parameter.edit(file=file, param="algorithm", newval="DRL")
    else:
        print("Please check the policy input, fitted_Q or DRL. fitted_Q will be set as default.")
        Parameter.edit(file=file, param="algorithm", newval="fitted_Q")
    Parameter.readfile(file=file)
    print("Learning algorithm / Policy: " + Parameter.algorithm)
    RandomSamplingForPSR = True
    isbuiltPSR = True
    game = PacMan()
    game.calulateMaxTestID()
    Parameter.maxTestID = game.maxTestID
    trainData = TrainingData()
    iterNo = 0
    agent = Agent(PnumActions=game.getNumActions(), epsilon=Parameter.epsilon,
                  inputDim=(Parameter.svdDim,), algorithm=Parameter.algorithm, Parrallel=True)
    rdict = readMemoryfromdisk(file="PSR/rewardDict.txt")
    copyRewardDict(rewardDict=rewardDict, rewardDict1=rdict)
    psrModel = CompressedPSR(game.getGameName())
    psrPool = Pool(Parameter.threadPoolSize, initializer=init, initargs=(Parameter.maxTestID, file, Lock(),))
    print("Finishing Preparation!")
    loadCheckPoint(trainData=trainData, epoch=iterNo, psrModel=psrModel, rewardDict=rewardDict)
    trainData = trainData.MergeAllBatchData()
    trainSet = None
    while iterNo < epochs:
        print("Starting Iteration: " + str(iterNo + 1))
        if RandomSamplingForPSR:
            trainData.newDataBatch()
            game.SimulateTrainData(runs=Parameter.runsForCPSR, isRandom=True, psrModel=psrModel,
                                   trainData=trainData, epoch=iterNo - 1, pool=psrPool,
                                   RunOnVirtualEnvironment=False, name=game.getGameName(), rewardDict=rewardDict,
                                   ns=ns)
            psrModel.validActObset = trainData.validActOb
            WriteEvalUateDataForPacMan(EvalData=trainData.data[trainData.getBatch()], epoch=-1)
            trainData.WriteData(file=dir + "/RandomSampling" + str(iterNo) + ".txt")
            RandomSamplingForPSR = False
        if isbuiltPSR:
            psrModel.build(data=trainData, aos=trainData.validActOb, pool=psrPool, rewardDict=rewardDict)
        psrModel.saveModel(epoch=iterNo)
        from bin.Util import writeMemoryintodisk
        writeMemoryintodisk(file="PSR/rewardDict.txt", data=rewardDict.copy())
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
        print("The time spent on training: " + str(tick2 - tick1) + "s")
        agent.SaveWeight(epoch=iterNo)
        print("Evaluating the agent")
        tick3 = time.time()
        EvalData = game.SimulateTestingRun(runs=Parameter.testingRuns, epoch=iterNo, pool=psrPool,  #edit
                                           psrModel=psrModel, name=game.getGameName(), rewardDict=rewardDict, ns=ns) #edit
        tick4 = time.time()
        print("The time spent on evaluating agent: " + str(tick4 - tick3) + "s")
        trainData.newDataBatch()
        game.SimulateTrainData(runs=Parameter.runsForLearning, psrModel=psrModel, trainData=trainData,
                               isRandom=False, epoch=iterNo, pool=psrPool,
                               RunOnVirtualEnvironment=Parameter.trainingOnVirtualEnvironment,
                               name=game.getGameName(), rewardDict=rewardDict, ns=ns)
        trainData.WriteData(file=dir + "/observations/epsilonGreedySampling" + str(iterNo) + ".txt")
        WriteEvalUateDataForPacMan(EvalData=EvalData, epoch=iterNo)
        iterNo = iterNo + 1

if __name__ == "__main__":
    train(epochs=10, policy='fitted_Q')