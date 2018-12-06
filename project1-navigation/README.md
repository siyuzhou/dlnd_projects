# Navigation

## 1. Environment

The Banana environment consists of an enclosed rectangular plane with yellow and blue bananas placed randomly in it. The agent's observation of the state has `37` dimensions which contains the agent's velocity and ray-based forward perception of objects in agent's field of view. The agent has four available actions: `0` for forward, `1` for backward, `2` for left turn, and `3` for right turn. The agent receives a reward of `+1` for collecting a yellow banana, and `-1` for collecting a blue banana.

Download the environment from one of the links below and place it in the project folder.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## 2. Package dependencies

- NumPy
- [PyTorch](https://pytorch.org/)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)

For NumPy, simply run `pip install numpy` in your python environment. Follow installation instructions in the links for other packages.

## 3. Getting started

Checkout out the Jupyter notebook `Navigation.ipynb` for detailed information of the environment. Run through Section 4 to see how the agent is trained. Section demonstrate the trained agent in action!

Details of model implementation are in `dqn_agent.py` and `model.py`. Code modified from course examples.

