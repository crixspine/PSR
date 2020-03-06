from bin.ASimulator import Simulator
import numpy as np
from numpy.random import choice
from bin import Parameter


class Tiger95(Simulator):
    NUMActions = 3
    NUMObservations = 2
    NUMRewards = 3
    NUMStates = 2
    Rewards = dict()
    Rewards[0] = -100.0
    Rewards[1] = 10.0
    Rewards[2] = -1.0
    Actions = ["Open-Left", "Open-Right", "Listen"]
    Observations = ["Tiger-Left", "Tiger-Right"]
    States = ["Tiger-Left", "Tiger-Right"]
    StatesID = [0, 1]
    ObservationsID = [0, 1]

    TMatsOpen = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
    TMatsListen = np.array([[1.0, 0.0],
                            [0.0, 1.0]])
    TMats = dict()
    TMats[0] = TMatsOpen
    TMats[1] = TMatsOpen
    TMats[2] = TMatsListen

    OMatsOpen = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
    OMatsListen = np.array([[0.85, 0.15],
                            [0.15, 0.85]])
    OMats = dict()
    OMats[0] = OMatsOpen
    OMats[1] = OMatsOpen
    OMats[2] = OMatsListen

    RMatsOpenLeft = np.array([-100, 10])
    RMatsOpenRight = np.array([10, -100])
    RMatsListen = np.array([-1, -1])
    RMats = dict()
    RMats[0] = RMatsOpenLeft
    RMats[1] = RMatsOpenRight
    RMats[2] = RMatsListen
    Belief = [0.5, 0.5]

    def Clone(self):
        return Tiger95()

    def getRewardDict(self):
        return Tiger95.Rewards

    def getNumActions(self):
        return Tiger95.NUMActions

    def getNumObservations(self):
        return Tiger95.NUMObservations

    def getNumRewards(self):
        return Tiger95.NUMRewards

    def __init__(self):
        super().__init__()
        self.reward = None
        self.observation = None
        self.BatchNum = Parameter.SimulateBatch
        self.TigerBatch = None
        self.InitRun()

    def InitRun(self):
        self.TigerBatch = list(choice(a=Tiger95.StatesID, p=Tiger95.Belief, size=self.BatchNum))

    def getGameName(self):
        return "Tiger95"

    def executeAction(self, aid):
        if len(aid) != self.BatchNum:
            Exception("The threadNum are changed!")

        TMat = [Tiger95.TMats[a] for a in aid]
        OMat = [Tiger95.OMats[a] for a in aid]
        RMat = [Tiger95.RMats[a] for a in aid]
        self.reward = np.array([RMat[i][self.TigerBatch[i]] for i in range(len(RMat))])
        for key in Tiger95.Rewards.keys():
            self.reward[self.reward == Tiger95.Rewards[key]] = key
        # Taking Transition
        ps = [TMat[i][self.TigerBatch[i]] for i in range(len(TMat))]
        NewTiger = []
        for p in ps:
            NewTiger.append(choice(a=Tiger95.StatesID, p=list(p), size=1)[0])
        self.TigerBatch = NewTiger
        # Taking Observation
        p1s = [OMat[i][self.TigerBatch[i]] for i in range(len(OMat))]
        self.observation = []
        for p1 in p1s:
            self.observation.append(choice(a=Tiger95.ObservationsID, p=list(p1), size=1)[0])
        # if aid < 2:
        #     return True
        # return False

    def getObservation(self):
        o = self.observation[:]
        self.observation = None
        if o is None:
            Exception("Tiger95 doesn't generate new observation!")
        return o

    def getReward(self):
        r = self.reward[:]
        self.reward = None
        if r is None:
            Exception("Tiger95 doesn't generate new reward!")
        return r
