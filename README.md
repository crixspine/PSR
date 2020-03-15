# Predictive State Representation (PSR) Models

## Description
This toolkit allows the PSR modelling of dynamical systems to train agents in certain environments. The original code is from: *https://github.com/williamleif/PSRToolbox*, and is packaged into this modular Python library format which is much more usable.

## Basics
* **environment**: Provides all the environments available to train the agent in. ``Maze``, ``PacMan``, ``StandTiger`` and ``Tiger95`` are all implmented in the code itself. ``GymEnv`` uses the OpenAI gym environments: *https://gym.openai.com/*. Use only Atari games with RAM as input: *https://gym.openai.com/envs/#atari*
* **train**: Interface to train agent in their respective environments.
* **autoencoder**: This is only for using OpenAI Gym environments. An autoencoder is used to encode observations from Gym. `SimpleAutoEnc` is a simple autoencoder network. `DeepAutoEnc` is a deep autoencoder network with a hidden layer of neurons.

## Installation
Simply clone or download this repository and run the code as below.

## Running the code
Firstly, create a .py file in the same directory the repository is cloned into.

### To run in local Pacman environment:
```
import PSR.train.PacManTrain
if __name__ == "__main__":
    PSR.train.PacManTrain.train(epochs, policy)
```
Input Parameters:
- epochs: Training iterations
- policy: Learning algorithm, input **"fitted_Q"** or **"DRL"**

### To run in local Maze/StandTiger/Tiger95 environment:
```
import PSR.train.MazeTigerTrain
if __name__ == "__main__":
    PSR.train.MazeTigerTrain.train(game, epochs)
```
Input Parameters:
- game: Environment, input **"standtiger"**, **"tiger95"** or **"maze"**
- epochs: Training iterations

### To run in OpenAI Gym environment:
```
import PSR.train.GymTrain
if __name__ == "__main__":
    PSR.train.GymTrain.train(gameName, epochs, autoencoder, policy)
```
Input Parameters:
- gameName: Environment in OpenAI Gym, use only Atari games with RAM input, e.g. **"MsPacman-ram-v0"** or **"Boxing-ram-v0"**
- epochs: Training iterations
- autoencoder: Autoencoder network, input **"simple"** or **"deep"**
- policy: Learning algorithm, input **"fitted_Q"** or **"DRL"**

## Acknowledgements
Many thanks to @zhoujingzhe for translating the original code into Python.
