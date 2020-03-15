from bin.ASimulator import Simulator
import numpy as np
from numpy.random import choice
from bin import Parameter

class Maze(Simulator):
    NUMActions = 5
    NUMObservations = 6
    NUMRewards = 3
    NUMStates = 11
    Rewards = dict()
    Rewards[0] = -0.04
    Rewards[1] = 10.0
    Rewards[2] = -100.0
    Actions = ["Move-north", "Move-south", "Move-east", "Move-west", "Reset"]
    Observations = ["left", "right", "neither", "both", "good", "bad"]
    States = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    StatesID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ObservationsID = [0, 1, 2, 3, 4, 5]
    Belief = [0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111]
    Move_North = np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.1],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.1]])

    Move_South = np.array([[0.1, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]])

    Move_East = np.array([[0.1, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9]])

    Move_West = np.array([[0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1]])

    l = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.111111, 0.111111, 0.111111, 0.0, 0.111111, 0.111111, 0.0, 0.111112, 0.111111, 0.111111, 0.111111],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    TMats = dict()
    TMats[0] = Move_North
    TMats[1] = Move_South
    TMats[2] = Move_East
    TMats[3] = Move_West
    TMats[4] = l

    Ob = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    Reset = np.array([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]])
    OMats = dict()
    OMats[0] = Ob
    OMats[1] = Reset

    R1 = [-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]
    R2 = [-0.04, -0.04, -0.04, 10.0, -0.04, -0.04, -100.0, -0.04, -0.04, -0.04, -0.04]
    RMats = dict()
    RMats[0] = R1
    RMats[1] = R2

    def Clone(self):
        return Maze()

    def getRewardDict(self):
        return Maze.Rewards

    def getNumActions(self):
        return Maze.NUMActions

    def getNumObservations(self):
        return Maze.NUMObservations

    def getNumRewards(self):
        return Maze.NUMRewards

    def __init__(self):
        super().__init__()
        self.InitRun()
        self.rewards = None
        self.observations = None
        self.agents = None

    def InitRun(self):
        self.agents = list(choice(a=Maze.StatesID, p=Maze.Belief, size=Parameter.SimulateBatch))

    def getGameName(self):
        return "Maze"

    def executeAction(self, aid):
        aids = aid
        TMats = [Maze.TMats[aid] for aid in aids]
        OMats = []
        RMats = []
        for aid in aids:
            if aid != 4:
                OMats.append(Maze.OMats[0])
                RMats.append(Maze.RMats[0])
            else:
                OMats.append(Maze.OMats[1])
                RMats.append(Maze.RMats[1])
        # Taking Reward
        rewards = np.array([RMats[idx][agent] for idx, agent in zip(range(Parameter.SimulateBatch), self.agents)])
        for key in Maze.Rewards.keys():
            rewards[rewards == Maze.Rewards[key]] = key
        self.rewards = rewards.astype(int)
        # Taking Transition
        ps = [TMats[idx][agent] for idx, agent in zip(range(Parameter.SimulateBatch), self.agents)]
        newAgents = []
        for p in ps:
            newAgents.append(choice(a=Maze.StatesID, p=list(p), size=1)[0])
        self.agents = newAgents
        # Taking Observation
        os = [OMats[idx][agent] for idx, agent in zip(range(Parameter.SimulateBatch), self.agents)]
        self.observations = [choice(a=Maze.ObservationsID, p=list(o), size=1)[0] for o in os]

    def getObservation(self):
        o = self.observations[:]
        self.observations = None
        if o is None:
            Exception("Maze doesn't generate new observation!")
        return o

    def getReward(self):
        r = self.rewards[:]
        self.rewards = None
        if r is None:
            Exception("Maze doesn't generate new reward!")
        return r