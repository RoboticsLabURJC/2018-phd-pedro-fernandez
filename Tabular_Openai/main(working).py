# main.py
 
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

 
# Using the gym library to create the environment
env = gym.make('MountainCar-v0')

#print(env.action_space.n)
#print(len(env.observation_space.high))
#print(env.observation_space.n)
#print(env.observation_space.high)
#print(env.observation_space.low)
env.reset()
 
params = Params(env)

#-------------------------------
# Defining all the required parameters
#EPISODES = 100
#LEARNING_RATE = 0.1 # alpha: min 0 - max 1
#DISCOUNT = 0.95 # gamma: min 0 - max 1

#REWARD_END = 0
#MAX_STEPS = 200
#-------------------------------
# Exploration settings
#EPSILON = 0.5  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL
#START_EPSILON_DECAYING = 1
#END_EPSILON_DECAYING = EPISODES #we can play with this var
#epsilon_decay_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
#epsilon_decay_value = 0.004

#epsilon = 0.1
#total_episodes = 500
#max_steps = 100
#alpha = 0.5
#gamma = 1

#-------------------------------
# we have to discretize continuum variables. We have to find the optimal Qtable size for the problem
#DISCRETE_OS_SIZE = [20, 20]
#DISCRETE_VALUE = 40
#DISCRETE_OS_SIZE = [DISCRETE_VALUE] * len(env.observation_space.high)
#discrete_os_win_size = (env.observation_space.high -
#                        env.observation_space.low)/DISCRETE_OS_SIZE

#DISCRETE = True

# create QTable 
#q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
q_table = np.random.uniform(low=-2, high=0, size=(params.DISCRETE_OS_SIZE + [params.env.action_space.n]))


#------------------------------
#to convert continuos states into discrete states. So we can index QTable with discrete_state return value: q_table[discrete_state] to get the values in those positions
def get_discrete_state(state):
    discrete_state = (state - params.env.observation_space.low) / params.discrete_os_win_size
    return tuple(discrete_state.astype(int))



"""
    The next parameters below is used to calculate
    the reward by each algorithm
"""
#for stats
#SHOW_EVERY = 200
#MIN_VALUE = -200 # -200 for mountainCar

# Save Rewards in each episode
#STATS_EVERY = 100
#SAVE_TABLE_EVERY = 1000

ep_rewards = []
aggr_ep_rewards = {
    'ep': [],
    'avg': [],
    'max': [],
    'min': []}


episodeReward = 0
totalReward = {
    'SarsaAgent': [],
    'QLearningAgent': [],
    'ExpectedSarsaAgent': []
}
 
# Defining all the three agents
#expectedSarsaAgent = ExpectedSarsaAgent(
#    epsilon, alpha, gamma, env.observation_space.n,
#    env.action_space.n, env.action_space)

print(f"EPSILON: {params.EPSILON}, LEARNING_RATE: {params.LEARNING_RATE}, DISCOUNT: {params.DISCOUNT}, DISCRETE_VALUE: {params.DISCRETE_VALUE}, env.observation_space.high: {params.env.observation_space.high},\
    env.action_space.n: {params.env.action_space.n}, env.action_space: {params.env.action_space}, \
        DISCRETE: {params.DISCRETE}")

qLearningAgent = QLearningAgent(
    q_table, params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
#sarsaAgent = SarsaAgent(
#    epsilon, alpha, gamma, env.observation_space.n,
#    env.action_space.n, env.action_space)
 
# Now we run all the episodes and calculate the reward obtained by
# each agent at the end of the episode
 
#agents = [expectedSarsaAgent, qLearningAgent, sarsaAgent]
agents = [qLearningAgent]
 
for agent in agents:
    for episode in range(params.EPISODES + 1):
        # Initialize the necessary parameters before
        # the start of the episode
        #t = 0
        #state1 = env.reset()
        #action1 = agent.choose_action(state1)
        #episodeReward = 0 #este es el antiguo
        episode_reward = 0 #este es de Harrison

        if episode % params.SHOW_EVERY == 0:
            print("episode:", episode)
            render = True
        else:
            render = False 

        discrete_state = get_discrete_state(params.env.reset())
        #action1 = agent.choose_action(discrete_state1)
        #print("discrete_state1:", discrete_state1)
        #print("action1:", action1)

        done = False
        while not done:
        #while t < MAX_STEPS:

            #################### here we can change CLASSES #########################
            #action = agent.choose_action(discrete_state)
            #########################################################################
            #action = agent.choose_action_inside(discrete_state)
            #########################################################################

            #print("action1:", action1)
            if np.random.random() > params.EPSILON: #IF EPSILON = 1, then always RANDOM actions (EXPLORATORY)
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
               # Get random action
                action = np.random.randint(0, params.env.action_space.n)

            #########################################################################
            #print(f"action: {action} y action_1: {action_1} y action_2: {action_2}")
            # Getting the next state, reward, and other parameters
            new_state, reward, done, info = params.env.step(action)
            episode_reward += reward

            new_discrete_state = get_discrete_state(new_state)
            #print(f"state2: {state2}, new_discrete_state: {new_discrete_state}")

            #env.render
            #if render: #no esta funcionando bien del todo la visualazion
                #    env.render()

            # Choosing the next action

            #action2 = agent.choose_action(new_discrete_state)

            if not done:
                #print(f"discrete_state1: {discrete_state1}, new_discrete_state: {new_discrete_state}, reward: {reward}, action1: {action1}, action2: {action2}")

                # Learning the Q-value
                #########################################################################
                #agent.update(discrete_state, new_discrete_state, reward, action)
                #########################################################################
                
                
                #print(agent.Q_table)
                #max_future_q = np.max(q_table[new_discrete_state])
                #print("np.max(q_table[new_discrete_state]", np.max(q_table[new_discrete_state]))
                    #print("q_table[new_discrete_state]:", q_table[new_discrete_state])
                #print("new_discrete_state:", new_discrete_state)

                # Current Q value (for current state and performed action)
                #current_q = q_table[discrete_state1 + (action1,)]
                    #print("discrete_state:", discrete_state)
                    #print("action:", action)
                    #print("discrete_state + (action,):", discrete_state + (action,))
                    #print("current_q", current_q)

                # And here's our equation for a new Q value for current state and action
                #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
                #        (reward + DISCOUNT * max_future_q)
                    #print("new_q:", new_q)

                # Update Q table with new Q value
                #q_table[discrete_state1 + (action1,)] = new_q

                #########################################################################    
                q_table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * q_table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + params.DISCOUNT * np.max(q_table[new_discrete_state]))
                #########################################################################

            elif new_state[0] >= params.env.goal_position:
                print(f"finish in episode {episode}")
                print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {params.EPSILON:>1.2f}, discount: {params.DISCOUNT:>2.2f},Learning Rate: {params.LEARNING_RATE}')



            #state1 = state2
            discrete_state = new_discrete_state
            #action1 = action2
             

            # Updating the respective vaLues
            #t += 1
            #episodeReward += reward
            #episode_reward += reward
             
            # If at the end of learning process
            #if done:
            #    break

            #elif new_state[0] >= env.goal_position:
            #    print(f"finish in episode {episode}")
                #q_table[discrete_state + (action,)] = reward
                #print(f"discrete_state (initial): {discrete_state} and q_table[discrete_state + (action,)]: {q_table[discrete_state + (action,)]}, and new_discrete_state: {new_discrete_state}")
            #    print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}, discount: {discountrates:>2.2f},Learning Rate: {learningrates}')

            #    q_table[discrete_state + (action,)] = REWARD_END #0
            #    agent.update(discrete_state1, new_discrete_state, REWARD_END, action1, action2)

        if params.END_EPSILON_DECAYING >= episode >= params.START_EPSILON_DECAYING:
            #print(f'Entramos >----- current epsilon: {epsilon:>1.3f},epsilon_decay_value: {epsilon_decay_value:>1.5f}')
            #params.EPSILON -= params.epsilon_decay_value
            params.set_epsilon(params.epsilon_decay_value)
            #print(f"salgo params.EPSILON: {params.EPSILON}, episode: {episode}")
            #print(f'current epsilon: {EPSILON:>1.3f},epsilon_decay_value: {epsilon_decay_value:>1.5f}')
        #else:
        #    EPSILON = 0.01    

        # show some STATS 
        ep_rewards.append(episode_reward)
        #episodeReward.append(episode_reward)



        # Append the sum of reward at the end of the episode
        totalReward[type(agent).__name__].append(episode_reward)
 
        if not episode % params.STATS_EVERY:
            #print(f"entro aqui")
            #average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
            average_reward = sum(ep_rewards[-params.STATS_EVERY:])/len(ep_rewards[-params.STATS_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-params.STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-params.STATS_EVERY:]))
            print(f'Learning Rate: {params.LEARNING_RATE}, discount: {params.DISCOUNT:>2.2f}, episode: {episode:>5d}, average reward: {average_reward:>4.1f}\
,min reward: {min(ep_rewards[-params.STATS_EVERY:])}, max reward: {max(ep_rewards[-params.STATS_EVERY:])}\
,current epsilon: {params.EPSILON:>1.3f},epsilon_decay_value: {params.epsilon_decay_value:>1.5f}')

            if params.MIN_VALUE < max(ep_rewards[-params.STATS_EVERY:]):
                params.set_min_value(max(ep_rewards[-params.STATS_EVERY:]))
                #params.MIN_VALUE = max(ep_rewards[-params.STATS_EVERY:])
                #print(f"salgo de MIN_VALUE: {params.MIN_VALUE}")
            # lets save results in a table
            if episode % params.SAVE_TABLE_EVERY == 0 and episode > 0:
                print(f"episode: {episode}")
                #np.save(f"qtables/{agent}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", agent.Q_table)
                #path_name_table = os.path.join(agent,'-', LEARNING_RATE,DISCOUNT,episode,qtable.npy)
                #os.makedirs("Tables", exist_ok = True)
                #########################################################################
                #np.save(f"Tables/{type(agent).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", q_table)
                #np.save(f"Tables/{type(agent).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", agent.Q_table)
                agent.tables_rewards(params.LEARNING_RATE, params.DISCOUNT, episode, q_table)

    

    agent.draw_rewards(aggr_ep_rewards, params.LEARNING_RATE, params.DISCOUNT, episode, params.MIN_VALUE)


params.env.close()

#########################################################################
# Calculate the mean of sum of returns for each episode
meanReturn = {
    'SARSA-Agent': np.mean(totalReward['SarsaAgent']),
    'Q-Learning-Agent': np.mean(totalReward['QLearningAgent']),
    'Expected-SARSA-Agent': np.mean(totalReward['ExpectedSarsaAgent'])
}
 
bestReturn = {
    'SARSA-Agent': np.min(aggr_ep_rewards['min']),
    'Q-Learning-Agent': params.MIN_VALUE, #np.min(aggr_ep_rewards['min']),
    'Expected-SARSA-Agent': np.min(aggr_ep_rewards['min'])
}

# Print the results
print("------------------------------")
print(f"SARSA Average Sum of Reward: {meanReturn['SARSA-Agent']}")
print(f"Q-Learning Average Sum of Return: {meanReturn['Q-Learning-Agent']}")
print(f"Expected Sarsa Average Sum of Return: {meanReturn['Expected-SARSA-Agent']}")
print("------------------------------")
print(f"SARSA Average Sum of Reward: {bestReturn['SARSA-Agent']}")
print(f"Q-Learning Average Sum of Return: {bestReturn['Q-Learning-Agent']}")
print(f"Expected Sarsa Average Sum of Return: {bestReturn['Expected-SARSA-Agent']}")

agent.set_stats((params.MIN_VALUE))