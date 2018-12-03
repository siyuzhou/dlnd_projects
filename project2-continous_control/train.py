from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment
from agent import DDPGAgent

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(
    states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# Create agents
agents = [DDPGAgent(state_size=state_size, action_size=action_size, random_seed=10)]


def train(n_episodes=2000, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agents[0].reset()
        score = 0
        for t in range(max_t):
            action = agents[0].act(state)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agents[0].step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        if score > max_score:
            max_score = score

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}, Max Score: {:.2f}'.format(
            i_episode, np.mean(scores_deque), score, max_score), end="")
        if i_episode % 100 == 0:
            torch.save(agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}, Max Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque), max_score))

    return scores


scores = train()

np.save("score.npy", scores)

env.close()
