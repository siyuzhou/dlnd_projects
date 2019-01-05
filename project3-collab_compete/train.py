import os
import sys
import argparse
from collections import deque, namedtuple

import numpy as np
import torch
from unityagents import UnityEnvironment
from agent import DDPGAgent


def save_checkpoint(agent, total_episode, logdir):
    torch.save(agent.actor_local.state_dict(), os.path.join(logdir, "actor_local.pth"))
    torch.save(agent.actor_target.state_dict(), os.path.join(logdir, "actor_target.pth"))
    torch.save(agent.critic_local.state_dict(), os.path.join(logdir, "critic_local.pth"))
    torch.save(agent.critic_target.state_dict(), os.path.join(logdir, "critic_target.pth"))

    torch.save(total_episode, os.path.join(logdir, "episodes.pth"))


def load_checkpoint(logdir):
    Checkpoint = namedtuple("Checkpoint",
                            ['actor_local',
                             'actor_target',
                             'critic_local',
                             'critic_target',
                             'episodes'])
    try:
        actor_local = torch.load(os.path.join(logdir, "actor_local.pth"))
        actor_target = torch.load(os.path.join(logdir, "actor_target.pth"))
        critic_local = torch.load(os.path.join(logdir, "critic_local.pth"))
        critic_target = torch.load(os.path.join(logdir, "critic_target.pth"))
        episodes = torch.load(os.path.join(logdir, "episodes.pth"))
        return Checkpoint(actor_local, actor_target, critic_local, critic_target, episodes)
    except FileNotFoundError:
        if not os.path.exists(logdir):
            os.mkdir(logdir)


def train(agent, n_episodes, max_t, logdir):
    checkpoint = load_checkpoint(logdir)
    total_episode = 0
    if checkpoint:
        agent.load(checkpoint)
        total_episode += checkpoint.episodes

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        total_episode += 1

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.mean(reward)
            if all(done):
                break

        max_score = max(score, max_score)

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tMax Score: {:.2f}'.format(
            total_episode, np.mean(scores_deque), score, max_score), end="")
        sys.stdout.flush()

        if i_episode % 100 == 0:
            save_checkpoint(agent, total_episode, logdir)

            print('\rEpisode {}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(
                total_episode, np.mean(scores_deque), max_score))

    return scores


if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-t", type=int, default=1000)
    parser.add_argument("--logdir", type=str, default="checkpoint")
    ARGS = parser.parse_args()

    # Create environment.
    env = UnityEnvironment(file_name='environment/Tennis_Linux/Tennis.x86')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    # Create agent
    agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=10)

    scores = train(agent, ARGS.episodes, ARGS.max_t, ARGS.logdir)

    np.save("scores.npy", scores)
