import sys
import os



from PSR.bin import Agent, ASimulator, MultiProcessSimulation, Parameter, TrainingData, Util
from PSR.environment import GymEnv, Maze, PacMan, StandTiger, Tiger95
from PSR.model import PSRmodel, TPSR
from PSR.train import PacManTrain, StandTigerTrain