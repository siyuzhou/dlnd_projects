# Continuous Control

## 1. Environment

The environment contains a moving ball and a two joints robotic arm that attempts to reach the ball. The agent receives 0.1 reward for every step the arm successfully reaches the goal. 

The observational space consists of 33 dimentional vector which corresponds to the position and velocity of the arm. Each action is a vector of 4 continuous numbers related to the torque applied to the arm. Each value is bound in the range [-1, 1] The environment is considered solved when an average score of 30+ is achieved over 100 consecutive episodes.

Download the environment from one of the links below and place it in the project folder.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## 2. Package dependencies

- NumPy
- [PyTorch](https://pytorch.org/)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

For NumPy, simply run `pip install numpy` in your python environment. Follow installation instructions in the links for other packages.

## 3. Getting started

Checkout out the Jupyter notebook `Continuous_Control.ipynb` for detailed information of the environment. Run through Section 4 to resume training. The output would show the plateaued performance.

Details of model implementation are in `agent.py` and `model.py`. Code modified from course examples.

Model checkpoints are saved in the `checkpoint` folder, including 5 files: `actor_local.pth`, `actor_target.pth`, `critic_local.pth`, `critic_target.pth`, `episodes.pth`. The records for target networks and past number of episodes are necissary for resumption of training. Please check out the `load_checkpoint` function in the `train.py` script to see how it is structures.
