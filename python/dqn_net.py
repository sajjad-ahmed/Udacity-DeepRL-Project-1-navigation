from unityagents import UnityEnvironment
import numpy as np

import os

from collections import deque
import matplotlib.pyplot as plt

import torch


def dqn_network(max_t=1000, episode_num=2000, eps_start=1.0, eps_end=0.01, decay=0.995):
    _epsd = eps_start
    scores = []
    scores_mean = []
    scores_window = deque(maxlen=100)
    for i in range(1, episode_num + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, _epsd)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        scores_window.append(score)
        scores_mean.append(np.mean(scores_window))

        print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}\tLR: {}'
              .format(i, scores_mean[-1], _epsd, agent.lr_scheduler.get_lr()), end="")

        _epsd = max(eps_end, decay * _epsd)

        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.4f}\tLR: {}'
                  .format(i, scores_mean[-1], _epsd, agent.lr_scheduler.get_lr()))
        if np.mean(scores_window) >= 13.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores, scores_mean


### Create and train the Agent

## Create the environment
env = UnityEnvironment(environment_path)

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Get the environment info
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

## Create the DQN Agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0, lr_decay=0.9999)

## Train the Agent
scores, mean = dqn_network(n_episodes=200, eps_start=0.10, max_t=300, eps_end=0.01, decay=0.987)

## Plot the scores
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(mean)), mean)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(('Score', 'Mean'), fontsize='xx-large')

plt.show()
