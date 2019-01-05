# Continuous Control

## 1. Environment

The environment contains two rackets to be controlled by two agents that bounce a ball over a net. An agent receives a reward of +0.1 if it manages to hit the ball over the net, or -0.01 if it lets the ball hit the ground or out of bounds. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  As the rackets move in a vertical 2D plane, two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

Download the environment from one of the links below and place it in the project folder.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## 2. Package dependencies

- NumPy
- [PyTorch](https://pytorch.org/)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

For NumPy, simply run `pip install numpy` in your python environment. Follow installation instructions in the links for other packages.

## 3. Getting started

Checkout out the Jupyter notebook `Tennis.ipynb` for detailed information of the environment. Run through Section 4 to resume training. The output would show the plateaued performance.

Details of model implementation are in `agent.py` and `model.py`. Code modified from course examples. Part of the implementation of prioritized experience replay was borrowed from [rlcode/per](https://github.com/rlcode/per/blob/master/prioritized_memory.py), with bugs fixed and design pattern altered.

Model checkpoints are saved in the `checkpoint` folder, including 5 files: `actor_local.pth`, `actor_target.pth`, `critic_local.pth`, `critic_target.pth`, `episodes.pth`. The records for target networks and past number of episodes are necissary for resumption of training. Please check out the `load_checkpoint` function in the `train.py` script to see how it is structures.
