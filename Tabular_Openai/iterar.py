
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem
from itertools import product

import gym
from Tabular_agents import ExpectedSarsaAgent, QLearningAgent, SarsaAgent, Params

from math import floor
from time import sleep
import os
import datetime as dt
import time


# Using the gym library to create the environment
env = gym.make('MountainCar-v0')
 
params = Params(env)
params.env.reset()

# Defining all the required parameters
epsilon = 0.1
#total_episodes = 500
max_steps = 200
alpha = 0.5
gamma = 1
"""
    The two parameters below is used to calculate
    the reward by each algorithm
"""
episodeReward = 0
totalReward = {
    'SarsaAgent': [],
    'QLearningAgent': [],
    'ExpectedSarsaAgent': []
}
 
# Defining all the three agents
qLearningAgent = QLearningAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)

sarsaAgent = SarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
expectedSarsaAgent = ExpectedSarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
##########################################
agents = {
    "Q-learning": QLearningAgent,
    "Expected Sarsa": ExpectedSarsaAgent
}
#env = cliffworld_env.Environment
#all_reward_sums = {}
step_sizes = np.linspace(0.1,1.0,10)
agent_info = {"num_actions": 4, "num_states": 48, "epsilon": 0.1, "discount": 1.0}
env_info = {}
num_runs = 10
num_episodes = 200
all_reward_sums = {}

#algorithms = ["Q-learning", "Expected Sarsa"]


# Now we run all the episodes and calculate the reward obtained by
# each agent at the end of the episode
 
agents = [expectedSarsaAgent, qLearningAgent, sarsaAgent]


cross_product = list(product(agents, step_sizes, range(num_runs)))
for agent, step_size, run in tqdm(cross_product):

        if (agent, step_size) not in all_reward_sums:
            all_reward_sums[(agent, step_size)] = []

        agent_info["step_size"] = step_size
        agent_info["run"] = run

        qLearningAgent = QLearningAgent(
            params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
            params.env.action_space.n, params.env.action_space, params.DISCRETE)

        sarsaAgent = SarsaAgent(
            params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
            params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
        expectedSarsaAgent = ExpectedSarsaAgent(
            params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
            params.env.action_space.n, params.env.action_space, params.DISCRETE)



        for _ in range(num_episodes):
        # Initialize the necessary parameters before
        # the start of the episode
            t = 0
            state1 = env.reset()
            action1 = agent.choose_action(state1)
            episodeReward = 0
            while t < max_steps:
    
                # Getting the next state, reward, and other parameters
                state2, reward, done, info = env.step(action1)
        
                # Choosing the next action
                action2 = agent.choose_action(state2)
                
                # Learning the Q-value
                agent.update(state1, state2, reward, action1, action2)
        
                state1 = state2
                action1 = action2
                
                # Updating the respective vaLues
                t += 1
                episodeReward += reward
                
                # If at the end of learning process
                if done:
                    break
            # Append the sum of reward at the end of the episode
            totalReward[type(agent).__name__].append(episodeReward)
            all_reward_sums[(agent, step_size)].append(episodeReward)


env.close()


for agent in agents:
    algorithm_means = np.array([np.mean(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
    algorithm_stds = np.array([sem(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
    plt.plot(step_sizes, algorithm_means, marker='o', linestyle='solid', label=type(agent).__name__)
    plt.fill_between(step_sizes, algorithm_means + algorithm_stds, algorithm_means - algorithm_stds, alpha=0.2)

plt.legend(loc=0)
plt.xlabel("Step-size")
plt.ylabel("Sum of\n rewards\n per episode",rotation=0, labelpad=50)
plt.xticks(step_sizes)
plt.show()
 
# Calculate the mean of sum of returns for each episode
meanReturn = {
    'SARSA-Agent': np.mean(totalReward['SarsaAgent']),
    'Q-Learning-Agent': np.mean(totalReward['QLearningAgent']),
    'Expected-SARSA-Agent': np.mean(totalReward['ExpectedSarsaAgent'])
}
 
# Print the results
print(f"SARSA Average Sum of Reward: {meanReturn['SARSA-Agent']}")
print(f"Q-Learning Average Sum of Return: {meanReturn['Q-Learning-Agent']}")
print(f"Expected Sarsa Average Sum of Return: {meanReturn['Expected-SARSA-Agent']}")