global numAtoms, learningRate, decay, vmin, vmax, gamma, alpha, trainingOnVirtualEnvironment, \
    epsilon, algorithm, svdDim, projDim, maxTestID, maxTestlen, maxHistLen, introduceReward, randomInit, \
    lengthOfAction, runsForLearning, runsForCPSR, testingRuns, SimulateBatch, threadPoolSize, lr_min, maxEpochs
import json

def readfile(file):
    with open(file, 'r') as f:
        Params = json.load(f)
        global numAtoms, learningRate, decay, vmin, vmax, gamma, alpha, trainingOnVirtualEnvironment,\
            epsilon, algorithm, svdDim, projDim, maxTestID, maxTestlen, maxHistLen, introduceReward, randomInit,\
            lengthOfAction, runsForLearning, runsForCPSR, testingRuns, SimulateBatch, threadPoolSize, lr_min, maxEpochs
        numAtoms = Params["numAtoms"]
        learningRate = Params["learningRate"]
        decay = Params["decay"]
        vmin = float(Params["vmin"])
        vmax = float(Params["vmax"])
        gamma = Params["gamma"]
        alpha = Params["alpha"]
        trainingOnVirtualEnvironment = Params["TrainingOnVirtualEnvironment"]
        epsilon = Params["epsilon"]
        algorithm = Params["algorithm"]
        maxEpochs = Params["maxEpoch"]
        ########################################################
        # PSR setting
        svdDim = Params["svdDim"]
        projDim = Params["ProjDim"]
        maxTestlen = Params["maxTestlen"]
        maxHistLen = Params["maxHistLen"]
        introduceReward = Params["introduceReward"]
        ########################################################
        randomInit = Params["RandomInit"]
        lengthOfAction = Params["LengthOfAction"]
        runsForLearning = Params["runsForLearning"]
        runsForCPSR = Params["runsForCPSR"]
        testingRuns = Params["TestingRuns"]
        maxTestID = Params["maxTestID"]
        threadPoolSize = Params["ThreadPoolSize"]
        lr_min = Params["minimum_learningRate"]

def writefile(file):
    Params = dict()
    Params["numAtoms"] = numAtoms
    Params["learningRate"] = learningRate
    Params["decay"] = decay
    Params["vmin"] = vmin
    Params["vmax"] = vmax
    Params["gamma"] = gamma
    Params["alpha"] = alpha
    Params["TrainingOnVirtualEnvironment"] = trainingOnVirtualEnvironment
    Params["epsilon"] = epsilon
    Params["algorithm"] = algorithm
    ########################################################
    # PSR setting
    Params["svdDim"] = svdDim
    Params["ProjDim"] = projDim
    Params["maxTestlen"] = maxTestlen
    Params["maxHistLen"] = maxHistLen
    Params["introduceReward"] = introduceReward
    ########################################################
    Params["RandomInit"] = randomInit
    Params["LengthOfAction"] = lengthOfAction
    Params["runsForLearning"] = runsForLearning
    Params["runsForCPSR"] = runsForCPSR
    Params["TestingRuns"] = testingRuns
    Params["maxTestID"] = maxTestID
    Params["ThreadPoolSize"] = threadPoolSize
    Params["minimum_learningRate"] = lr_min
    Params["maxEpoch"] = maxEpochs
    f = open(file=file, mode='w')
    json.dump(Params, f)
    f.close()

def edit(file, param, newval):
    with open(file, 'r+') as f:
        data = json.load(f)
        data[param] = newval
        f.seek(0)
        json.dump(data, f)
        f.truncate()