import os
import numpy as np
from bin import Parameter
from bin.MultiProcessSimulation import ConstructingTHMats
from bin.Util import writeMemoryintodisk, readaoMatsFromdisk, readMemoryfromdisk, writeDataintoDisk

eps = np.finfo(float).eps

class CompressedPSR:
    def TruncatedSVD(self, mats, maxDim):
        u, s, vh = np.linalg.svd(a=mats)
        numToKeep = min(len(s), maxDim)
        s = s[:numToKeep:]
        S = np.diag(s)
        U = u[:, :numToKeep:]
        Vh = vh[:numToKeep:, :]
        return (U, S, Vh)

    def LoadedExistedModel(self, file):
        import json
        with open(file) as json_file:
            data = json.load(json_file)
            self.pv = np.array(data['pv'])
            self.mInf = np.array(data['m_inf'])
            self.mInf = np.squeeze(a=self.mInf, axis=0)
            self.CaoMats = data['M_ao']
            for key in self.CaoMats.keys():
                self.CaoMats[key] = np.array(self.CaoMats[key])
            self.isbuilt = True

    def generateCoreTestPrediction(self, pvs):
        if self.CoreTests is None:
            return pvs
        Preds = self.PredictsForPV(pv1s=pvs)
        outputs = []
        for i in range(len(self.CoreTests)):
            outputs.append(Preds[self.CoreTests[i]])
        outputs = np.array(outputs)
        outputs = np.expand_dims(a=outputs, axis=1)
        return outputs

    def ReturnEmptyObject(self, name):
        psrModel = CompressedPSR(game=name)
        return psrModel

    def __init__(self, game):
        self.e = np.zeros((Parameter.projDim + 1, 1))
        self.e[0, 0] = 1
        self.randomVectorCache = dict()
        self.THMat = None
        self.HMat = None
        self.CaoMats = dict()
        self.aoMats = dict()
        self.isbuilt = False
        self.validActObset = None
        self.maxTestID = None
        self.CoreTests = None
        self.game = game

    def saveModel(self, epoch):
        dir = os.path.abspath(os.getcwd())
        if not os.path.exists(dir + "/observations/Epoch " + str(epoch)):
            os.makedirs(dir + "/observations/Epoch " + str(epoch))
        writeMemoryintodisk(file=dir + "/observations/Epoch " + str(epoch) + "/mInf.txt", data=self.mInf.tolist())
        writeMemoryintodisk(file=dir + "/observations/Epoch " + str(epoch) + "/pv.txt", data=self.pv.tolist())
        aoMats = dict()
        for key in self.CaoMats.keys():
            aoMats[key] = self.CaoMats[key].tolist()
        writeMemoryintodisk(file=dir + "/observations/Epoch " + str(epoch) + "/aoMats.txt", data=aoMats)

    def loadModel(self, epoch):
        dir = os.path.abspath(os.getcwd())
        if not os.path.exists(dir + "/observations/Epoch " + str(epoch)):
            os.makedirs(dir + "/observations/Epoch " + str(epoch))
        self.CaoMats = readaoMatsFromdisk(file=dir + "/observations/Epoch " + str(epoch) + "/aoMats.txt")
        self.mInf = np.array(readMemoryfromdisk(file=dir + "/observations/Epoch " + str(epoch) + "/mInf.txt"))
        self.pv = np.array(readMemoryfromdisk(file=dir + "/observations/Epoch " + str(epoch) + "/pv.txt"))
        self.isbuilt = True
        keys = self.CaoMats.keys()
        self.validActObset = list(keys)

    def build(self, data, aos, pool, rewardDict):
        import os
        dir = os.path.abspath(os.getcwd())
        # initalize multiprocess pool in each round in case the new data comes
        if Parameter.maxTestID == -1:
            Exception("maxTestID not been updated")
        self.validActObset = aos
        if self.THMat is None and self.HMat is None:
            print("Initialize THMat and HMat!")
            self.THMat = np.zeros((Parameter.projDim, Parameter.projDim + 1))
            print("THMat:")
            print(self.THMat)
            self.HMat = np.zeros((Parameter.projDim + 1, 1))
            print("HMat: ")
            print(self.HMat)

        for ao in self.validActObset:
            if ao not in self.aoMats.keys():
                self.aoMats[ao] = np.zeros((Parameter.projDim, Parameter.projDim + 1))
        actObsPerThread = int(len(data.data[data.getBatch()]) / Parameter.threadPoolSize)
        args = []
        for i in range(Parameter.threadPoolSize):
            d = data.data[data.getBatch()][i * actObsPerThread:(i + 1) * actObsPerThread:]
            fileName = dir + "/tmp/dataForThread" + str(i) + ".txt"
            writeDataintoDisk(file=fileName, data=d)
            tmpTrainData = data.ReturnEmptyObject()
            args.append([fileName, data.testDict, data.histDict, data.validActOb, "CPSR", i, tmpTrainData, rewardDict])
        outputs = pool.map(func=ConstructingTHMats, iterable=args)
        print("Constructing the TH aoMats is finished!")
        THMat = [output[0] for output in outputs]
        HistMat = [output[1] for output in outputs]
        files = [output[2] for output in outputs]
        aoMats = []
        import os
        for file in files:
            aoMats.append(readaoMatsFromdisk(file=file))
            os.remove(file)
        THMat = np.array(THMat)
        HistMat = np.array(HistMat)
        THMat = np.sum(a=THMat, axis=0)
        HistMat = np.sum(a=HistMat, axis=0)
        self.THMat = self.THMat + THMat
        print("THMat:")
        print(self.THMat)
        self.HMat = self.HMat + HistMat
        print("HMat: ")
        print(self.HMat)
        for ao in self.validActObset:
            for i in range(len(aoMats)):
                self.aoMats[ao] = self.aoMats[ao] + aoMats[i][ao]
        ret = self.TruncatedSVD(mats=self.THMat, maxDim=Parameter.svdDim)
        u = ret[0]
        s = ret[1]
        vT = ret[2]
        Z = u.transpose()
        pseudoInverse = np.linalg.pinv(np.matmul(Z, self.THMat))
        for ao in self.validActObset:
            self.CaoMats[ao] = np.matmul(np.matmul(Z, self.aoMats[ao]), pseudoInverse)
        PQ = np.matmul(Z, self.THMat)
        self.mInf = np.linalg.lstsq(a=PQ.transpose(), b=self.HMat)[0]
        self.pv = np.matmul(PQ, self.e)
        self.isbuilt = True

    def Starting(self, name):
        if name == "Tiger95":
            if Parameter.introduceReward:
                self.pv = self.PassAO(pvs=self.pv, aos="a0o0r0")
            else:
                self.pv = self.PassAO(pvs=self.pv, aos="a0o0")
        if name == "Maze":
            if Parameter.introduceReward:
                self.pv = self.PassAO(pvs=self.pv, aos="a4o0r0")
            else:
                self.pv = self.PassAO(pvs=self.pv, aos="a4o0")
        if name == "StandTiger":
            if Parameter.introduceReward:
                self.pv = self.PassAO(pvs=self.pv, aos="a1o0r2")
            else:
                Exception("StandTiger needs reward signal")

    def getCurrentPV(self):
        return self.pv[:]

    def UnseenAO(self, pv, AO):
        numerator = None
        for ao in self.CaoMats.keys():
            if ao[1] != AO[1]:
                continue
            aoMat = self.CaoMats[ao]
            if numerator is None:
                numerator = np.matmul(aoMat, pv)
            else:
                numerator = numerator + np.matmul(aoMat, pv)
        return np.round(a=numerator, decimals=5)

    def PassAO(self, pvs, aos):
        if aos not in self.CaoMats.keys():
            return self.UnseenAO(pv=pvs, AO=aos)
        aoMats = self.CaoMats[aos]
        numerators = np.matmul(aoMats, pvs)
        denumerators = self.PredictOneStepAO(pvs=pvs, aos=aos)
        pv1 = (numerators + eps) / (denumerators + eps)
        pv1 = np.round(a=pv1, decimals=5)
        return np.array(pv1)

    def PredictOneStepAO(self, pvs, aos):
        if not isinstance(pvs, np.ndarray):
            pvs = np.array(pvs)
        aoMats = self.CaoMats[aos]
        mInf = np.transpose(a=self.mInf, axes=[1, 0])
        likelihoodAO = np.matmul(np.matmul(mInf, aoMats), pvs)
        return likelihoodAO[0, 0]

    def PredictsForPV(self, pv1s):
        Predictions = dict()
        PredictionSum = dict()
        for ao in self.validActObset:
            a = ao[:2:]
            likelihood = self.PredictOneStepAO(pvs=pv1s, aos=ao)
            if likelihood < 0:
                likelihood = 0
            if a not in PredictionSum.keys():
                PredictionSum[a] = likelihood
            else:
                PredictionSum[a] = PredictionSum[a] + likelihood
            Predictions[ao] = likelihood
        for ao in self.validActObset:
            a = ao[:2:]
            try:
                Predictions[ao] = Predictions[ao] / (PredictionSum[a] + eps)
            except FloatingPointError:
                print(PredictionSum[a])
        return Predictions

    def Predicts(self, test):
        if Parameter.introduceReward:
            offset = 6
            idx = np.arange(0, len(test), 6, np.int)
        else:
            offset = 4
            idx = np.arange(0, len(test), 4, np.int)
        pv1 = self.pv[:]
        for i in idx:
            ao = test[i:i + offset:]
            pv1 = self.PassAO(pvs=pv1, aos=ao)
        Predictions = dict()
        PredictionSum = dict()
        for ao in self.validActObset:
            a = ao[:2:]
            likelihood = self.PredictOneStepAO(pvs=pv1, aos=ao)
            likelihood = likelihood
            if likelihood < 0:
                likelihood = 0
            if a not in PredictionSum.keys():
                PredictionSum[a] = likelihood
            else:
                PredictionSum[a] = PredictionSum[a] + likelihood
            Predictions[ao] = likelihood
        for ao in self.validActObset:
            a = ao[:2:]
            if PredictionSum[a] != 0:
                Predictions[ao] = Predictions[ao] / PredictionSum[a]
        return Predictions, self.generateCoreTestPrediction(pvs=pv1)