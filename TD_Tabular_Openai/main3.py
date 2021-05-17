# main.py


'''
Main3 stands for:
    - 3 algorithms: QLearning, SARSA and expected SARSA

Hyperparamans have to be change manually.

NOTES:
in this version, we discretizes continous state space in a table.

''' 


import gym
import numpy as np
 
from Params import Params
from ExpectedSarsa import ExpectedSarsaAgent
from QLearning import QLearningAgent
from Sarsa import SarsaAgent

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from math import floor
from time import sleep
import os
import datetime as dt
import time

from tqdm import tqdm
from scipy.stats import sem
from itertools import product
import operator
import csv
 
# Using the gym library to create the environment
env = gym.make('MountainCar-v0')
#env.reset()
 
params = Params(env)
params.env.reset()



'''
    Discretizing Tables. We can work with continuos values, but that way 
    we can build tables to show results. It is more convenient 
'''

#to convert continuos states into discrete states. So we can index QTable with discrete_state return value: q_table[discrete_state] to get the values in those positions
def get_discrete_state(state):
    discrete_state = (state - params.env.observation_space.low) / params.discrete_os_win_size
    return tuple(discrete_state.astype(int))

"""
    The next parameters below is used to calculate
    the reward by each algorithm
"""
totalReward = {
    'SarsaAgent': [],
    'QLearningAgent': [],
    'ExpectedSarsaAgent': []
}
 
bestReward = {
    'SarsaAgent': [],
    'QLearningAgent': [],
    'ExpectedSarsaAgent': []
} 

bestEpisode = {
    'SarsaAgent': [],
    'QLearningAgent': [],
    'ExpectedSarsaAgent': []
} 

"""
    Initialize differents Agents
    SARSA
    Expected SARSA
    QLearning
""" 

print(f"EPSILON: {params.EPSILON}, LEARNING_RATE: {params.LEARNING_RATE}, DISCOUNT: {params.DISCOUNT}, DISCRETE_VALUE: {params.DISCRETE_VALUE}, env.observation_space.high: {params.env.observation_space.high},\
    env.action_space.n: {params.env.action_space.n}, env.action_space: {params.env.action_space}, \
        DISCRETE: {params.DISCRETE}") 


qLearningAgent = QLearningAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)

sarsaAgent = SarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
expectedSarsaAgent = ExpectedSarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
#agents = [expectedSarsaAgent]
agents = [qLearningAgent, sarsaAgent, expectedSarsaAgent]
agent_Qlearning = [qLearningAgent]
agent_SARSA = [sarsaAgent]
agents_expectedSARSA = [expectedSarsaAgent]

'''
    Initialize loop

'''
start_time = time.time() 
for agent in agents:

    params.reset_Values()
    # Create table for all algorithms
    table = np.random.uniform(low=-2, high=0, size=(params.DISCRETE_OS_SIZE + [params.env.action_space.n]))
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    '''
            Init the EPISODES

    '''

    for episode in range(params.EPISODES + 1):
        # Initialize the necessary parameters before
        # the start of the episode

        episode_reward = 0

        if episode % params.SHOW_EVERY == 0:
            print("episode:", episode)
            render = True
        else:
            render = False 

        discrete_state = get_discrete_state(params.env.reset())

        done = False
        while not done:
        #while t < MAX_STEPS:

            #################### here we can change CLASSES #########################
            #action = agent.choose_action(discrete_state)
            #########################################################################
            #action = agent.choose_action_inside(discrete_state)
            #########################################################################

            if np.random.random() > params.EPSILON: #IF EPSILON = 1, then always RANDOM actions (EXPLORATORY)
                # Get action from table
                action = np.argmax(table[discrete_state])
            else:
               # Get random action
                action = np.random.randint(0, params.env.action_space.n)

            #########################################################################
            # Getting the next state, reward, and other parameters
            new_state, reward, done, info = params.env.step(action)

            new_discrete_state = get_discrete_state(new_state)

            #########################################################################
            #env.render
            #if render: #no esta funcionando bien del todo la visualazion
                #    env.render()
            #########################################################################
            
            if not done:
                
                # Learning the Q-value
                #########################################################################
                #agent.update(discrete_state, new_discrete_state, reward, action)
                #########################################################################
                if agent == qLearningAgent:
                    #print("entro QLearning")
                #########################################################################    
                    table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + params.DISCOUNT * np.max(table[new_discrete_state]))
                #########################################################################
                elif agent == sarsaAgent:
                    if np.random.random() > params.EPSILON: #IF EPSILON = 1
                        # Get action from table
                        action2 = np.argmax(table[new_discrete_state])
                    else:
                        # Get random action
                        action2 = np.random.randint(0, params.env.action_space.n) 

                    table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + params.DISCOUNT * np.max(table[new_discrete_state + (action2,)]))
                #########################################################################
                elif agent == expectedSarsaAgent:
                    expected_q = 0
                    q_max = np.max(table[new_discrete_state,:])
                    greedy_actions = 1 #not divide by 0
                    for i in range(params.env.action_space.n):
                        if table[new_discrete_state][i] -- q_max:
                            greedy_actions += 1

                    non_greedy_action_probability = params.EPSILON /   params.env.action_space.n
                    greedy_action_probability = ((1 - params.EPSILON) / greedy_actions) + non_greedy_action_probability

                    for i in range(params.env.action_space.n):
                        if table[new_discrete_state][i] -- q_max:
                            expected_q += table[new_discrete_state][i] * greedy_action_probability
                        else:
                            expected_q += table[new_discrete_state][i] * non_greedy_action_probability

                    table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + params.DISCOUNT * expected_q)        



                  

            elif new_state[0] >= params.env.goal_position:
                print(f"finish in episode {episode}")
                print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {params.EPSILON:>1.2f}, discount: {params.DISCOUNT:>2.2f},Learning Rate: {params.LEARNING_RATE}')
                table[discrete_state + (action,)] = params.REWARD_END 


            discrete_state = new_discrete_state

            episode_reward += reward
             
        # Get the BEST Value in all episodes
        if params.get_bestValue() < episode_reward:
                params.set_bestValue(episode_reward)
                params.set_bestEpisode(episode)

        # EPSILON DECAY for exploratory 
        if params.END_EPSILON_DECAYING >= episode >= params.START_EPSILON_DECAYING:
            params.set_epsilon(params.epsilon_decay_value)

        # show some STATS 
        ep_rewards.append(episode_reward)

        # Append the sum of reward at the end of the episode
        totalReward[type(agent).__name__].append(episode_reward)
 
        if not episode % params.STATS_EVERY:
            average_reward = sum(ep_rewards[-params.STATS_EVERY:])/len(ep_rewards[-params.STATS_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-params.STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-params.STATS_EVERY:]))
            print(f'Agent: {type(agent).__name__}, Learning Rate: {params.LEARNING_RATE}, discount: {params.DISCOUNT:>2.2f}, episode: {episode:>5d}, average reward: {average_reward:>4.1f},min reward: {min(ep_rewards[-params.STATS_EVERY:])}, max reward: {max(ep_rewards[-params.STATS_EVERY:])},current epsilon: {params.EPSILON:>1.3f},epsilon_decay_value: {params.epsilon_decay_value:>1.5f}')

            # Get the BEST Value in all episodes
#            if params.MIN_VALUE < max(ep_rewards[-params.STATS_EVERY:]):
#                agent.set_bestValue(max(ep_rewards[-params.STATS_EVERY:]))
#                agent.set_bestEpisode(episode)

            # lets save results in a table
            if episode % params.SAVE_TABLE_EVERY == 0 and episode > 0:
                agent.tables_rewards(params.LEARNING_RATE, params.DISCOUNT, episode, table)

    
    # save draws
    agent.draw_rewards(aggr_ep_rewards, params.LEARNING_RATE, params.DISCOUNT, episode, params.get_bestValue())
    bestReward[type(agent).__name__].append(params.get_bestValue())
    bestEpisode[type(agent).__name__].append(params.get_bestEpisode())
    
    #agent.set_bestValue((params.MIN_VALUE))

params.env.close()

#########################################################################
# Calculate the mean of sum of returns for each episode
meanReturn = {
    'SARSA-Agent': np.mean(totalReward['SarsaAgent']),
    'Q-Learning-Agent': np.mean(totalReward['QLearningAgent']),
    'Expected-SARSA-Agent': np.mean(totalReward['ExpectedSarsaAgent'])
}
 
bestReturn = {
    'SARSA-Agent': bestReward['SarsaAgent'],
    'Q-Learning-Agent': bestReward['QLearningAgent'], #np.min(aggr_ep_rewards['min']),
    'Expected-SARSA-Agent': bestReward['ExpectedSarsaAgent']
}

bestEpisode = {
    'SARSA-Agent': bestEpisode['SarsaAgent'],
    'Q-Learning-Agent': bestEpisode['QLearningAgent'], #np.min(aggr_ep_rewards['min']),
    'Expected-SARSA-Agent': bestEpisode['ExpectedSarsaAgent']
}


# Print the results
print("------------------------------")
print(f"SARSA Average Sum of Reward: {meanReturn['SARSA-Agent']}")
print(f"Q-Learning Average Sum of Return: {meanReturn['Q-Learning-Agent']}")
print(f"Expected Sarsa Average Sum of Return: {meanReturn['Expected-SARSA-Agent']}")
print("------------------------------")
print(f"SARSA best Reward: {bestReturn['SARSA-Agent']} in episode: {bestEpisode['SARSA-Agent']} ")
print(f"Q-Learning best Return: {bestReturn['Q-Learning-Agent']} in episode: {bestEpisode['Q-Learning-Agent']} ")
print(f"Expected Sarsa Average Sum of Return: {bestReturn['Expected-SARSA-Agent']} in episode: {bestEpisode['Expected-SARSA-Agent']} ")

#agent.set_stats((params.MIN_VALUE))