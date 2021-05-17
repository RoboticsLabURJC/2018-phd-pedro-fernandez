

'''
Main3x1 stands for:
    - 3 algorithms: QLearning, SARSA and expected SARSA
    - 2 hyperparameter: 
            * alpha or step size or Learning rate which runs 10 times (or more) for each algorithm
            * gamma or discount rate which can run 10 times (or more) for each algorithm 


Be careful!!: running takes you long long time

NOTES:
in this version, we discretizes continuous state space in a table.

'''
 
import gym
import numpy as np
 
from Agent import Agent 
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


'''
    next function shows figures between algorithms and LEARNING RATE or DISCOUNTS

'''

def draw_2Variables(agents, all_reward_sums, episodes, step_sizes):
    for agent in agents:
        algorithm_means = np.array([np.mean(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
        algorithm_stds = np.array([sem(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
        plt.plot(step_sizes, algorithm_means, marker='o', linestyle='solid', label=type(agent).__name__)
        plt.fill_between(step_sizes, algorithm_means + algorithm_stds, algorithm_means - algorithm_stds, alpha=0.2)

    plt.legend(loc=0)
    plt.xlabel("Step-size")
    plt.ylabel("Sum of\n rewards\n per episode",rotation=0, labelpad=50)
    plt.xticks(step_sizes)
       
    os.makedirs("Images", exist_ok = True)            
    plt.savefig(f"Images/allRewards-{episodes}-{time.strftime('%Y%m%d-%H%M%S')}.jpg", bbox_inches='tight')
    plt.clf()
        #plt.show()



"""
    Initialize differents Agents
    SARSA
    Expected SARSA
    QLearning
""" 

#print(f"EPSILON: {params.EPSILON}, LEARNING_RATE: {params.LEARNING_RATE}, DISCOUNT: {params.DISCOUNT}, DISCRETE_VALUE: {params.DISCRETE_VALUE}, env.observation_space.high: {params.env.observation_space.high},\
#    env.action_space.n: {params.env.action_space.n}, env.action_space: {params.env.action_space}, \
#        DISCRETE: {params.DISCRETE}") 


qLearningAgent = QLearningAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)

sarsaAgent = SarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
expectedSarsaAgent = ExpectedSarsaAgent(
    params.EPSILON, params.LEARNING_RATE, params.DISCOUNT, params.DISCRETE_VALUE, params.env.observation_space.high,
    params.env.action_space.n, params.env.action_space, params.DISCRETE)
 
"""
    The next parameters below is used to calculate
    the reward by each algorithm
"""
best_stats = {
    'QLearning': {
        'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []},
     'SARSA': {
        'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []},
    'expected SARSA': {
        'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []}
}

QLearning_stats = {
    'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []
}
SARSA_stats = {
    'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []
}
expectedSARSA_stats = {
    'episode': [], 'learning_rate': [], 'best_value': [], 'discount': []
}


all_reward_sums = {}
best_stepsize = {}

#agents = [qLearningAgent]
agents = [qLearningAgent, sarsaAgent, expectedSarsaAgent]
#cross_product = list(product(agents, params.LEARNING_RATES, range(params.EPISODES+1)))

'''
    Initialize loop

'''
start_time = time.time()
#for agent, step_size, run in tqdm(cross_product):
for agent in tqdm(agents):
    for step_size in tqdm(params.LEARNING_RATES):
        for discount in tqmd(params.DISCOUNTS):

            '''
                Initialize inside the loop for each agent
            '''

            if (agent, step_size, discount) not in all_reward_sums:
                all_reward_sums[(agent, step_size, discount)] = []

            if (agent, step_size, discount) not in best_stepsize:
                best_stepsize[(agent, step_size, discount)] = []  

            params.reset_Values()
            table = np.random.uniform(low=-2, high=0, size=(params.DISCRETE_OS_SIZE + [params.env.action_space.n]))
            ep_rewards = []
            aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
            

            '''
                Init the EPISODES

            '''

            for episode in range(params.EPISODES+1):
                # Initialize the necessary parameters before
                # the start of the episode

                episode_reward = 0

                if episode % params.SHOW_EVERY == 0:
                    #print("episode:", episode)
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
                            table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + discount * np.max(table[new_discrete_state]))
                        #########################################################################
                        elif agent == sarsaAgent:
                            if np.random.random() > params.EPSILON: #IF EPSILON = 1
                                # Get action from table
                                action2 = np.argmax(table[new_discrete_state])
                            else:
                                # Get random action
                                action2 = np.random.randint(0, params.env.action_space.n) 

                            table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + discount * np.max(table[new_discrete_state + (action2,)]))
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

                            table[discrete_state + (action,)] = (1 - params.LEARNING_RATE) * table[discrete_state + (action,)] + params.LEARNING_RATE * (reward + discount * expected_q)        


                    elif new_state[0] >= params.env.goal_position:
                        #print(f"finish in episode {episode}")
                        #print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {params.EPSILON:>1.2f}, discount: {params.DISCOUNT:>2.2f},Learning Rate: {params.LEARNING_RATE}')
                        table[discrete_state + (action,)] = params.REWARD_END 

                    discrete_state = new_discrete_state

                    episode_reward += reward
                    
                # Get the BEST Value in all episodes
                if params.get_bestValue() < episode_reward:
                        params.set_bestValue(episode_reward)
                        params.set_bestEpisode(episode)
                        params.set_bestLEARNING_RATE(step_size)
                        #params.set_bestDISCOUNT(discount)
                        best_stepsize[(agent, step_size)].append(episode_reward)
                        if agent == qLearningAgent:
                            QLearning_stats['episode'].append(episode)
                            QLearning_stats['learning_rate'].append(step_size)
                            QLearning_stats['best_value'].append(episode_reward)
                            QLearning_stats['discount'].append(discount)
   
                        elif agent == sarsaAgent:
                            SARSA_stats['episode'].append(episode)
                            SARSA_stats['learning_rate'].append(step_size)
                            SARSA_stats['best_value'].append(episode_reward)
                            SARSA_stats['discount'].append(discount)
                        else:      
                            expectedSARSA_stats['episode'].append(episode)
                            expectedSARSA_stats['learning_rate'].append(step_size)
                            expectedSARSA_stats['best_value'].append(episode_reward)
                            expectedSARSA_stats['discount'].append(discount)

                            
                # EPSILON DECAY for exploratory 
                if params.END_EPSILON_DECAYING >= episode >= params.START_EPSILON_DECAYING:
                    params.set_epsilon(params.epsilon_decay_value)

                # show some STATS 
                ep_rewards.append(episode_reward)

                # Append the sum of reward at the end of the episode
                #totalReward[type(agent).__name__].append(episode_reward)
                all_reward_sums[(agent, step_size)].append(episode_reward)

                if not episode % params.STATS_EVERY:
                    average_reward = sum(ep_rewards[-params.STATS_EVERY:])/len(ep_rewards[-params.STATS_EVERY:])
                    aggr_ep_rewards['ep'].append(episode)
                    aggr_ep_rewards['avg'].append(average_reward)
                    aggr_ep_rewards['max'].append(max(ep_rewards[-params.STATS_EVERY:]))
                    aggr_ep_rewards['min'].append(min(ep_rewards[-params.STATS_EVERY:]))
                    #print(f'Agent: {type(agent).__name__}, Learning Rate: {params.LEARNING_RATE}, discount: {params.DISCOUNT:>2.2f}, episode: {episode:>5d}, average reward: {average_reward:>4.1f},min reward: {min(ep_rewards[-params.STATS_EVERY:])}, max reward: {max(ep_rewards[-params.STATS_EVERY:])},current epsilon: {params.EPSILON:>1.3f},epsilon_decay_value: {params.epsilon_decay_value:>1.5f}')

                    # Get the BEST Value in all episodes
        #            if params.MIN_VALUE < max(ep_rewards[-params.STATS_EVERY:]):
        #                agent.set_bestValue(max(ep_rewards[-params.STATS_EVERY:]))
        #                agent.set_bestEpisode(episode)

                    # lets save results in a table
                    if episode % params.SAVE_TABLE_EVERY == 0 and episode > 0:
                        agent.tables_rewards(params.LEARNING_RATE, params.DISCOUNT, episode, table)


params.env.close()


##----------------------------------------- STATS ZONE -----------------------------------------

end_time = time.time() 
exec_time = end_time - start_time
print("------------- Stats for debug purposes -----------------")

print(f"Start time: {start_time}, end time: {end_time}, and exec time: {exec_time} in seconds") 


##---------- Drawing 
draw_2Variables(agents, all_reward_sums, params.EPISODES, params.LEARNING_RATES)


print("------------- Stats for debug purposes -----------------")
print(f"Q-Learning best Return: {QLearning_stats['best_value']} in episode: {QLearning_stats['episode']} with step size(learning rate): {QLearning_stats['learning_rate']} and discount(gamma): {QLearning_stats['discount']}")
print(f"SARSA best Return: {SARSA_stats['best_value']} in episode: {SARSA_stats['episode']} with step size(learning rate): {SARSA_stats['learning_rate']} and discount(gamma): {SARSA_stats['discount']}")
print(f"expected SARSA best Return: {expectedSARSA_stats['best_value']} in episode: {expectedSARSA_stats['episode']} with step size(learning rate): {expectedSARSA_stats['learning_rate']} and discount(gamma): {expectedSARSA_stats['discount']}")


# --------- Saving Results 
os.makedirs("Files", exist_ok = True)             
QLearning_file = open(f"Files/QLearning_stats-episodes({params.EPISODES})-alphas({len(params.LEARNING_RATES)})-gammas({len(params.DISCOUNTS)})-{time.strftime('%Y%m%d-%H%M%S')}.csv", "a")
SARSA_file = open(f"Files/SARSA_stats-episodes({params.EPISODES})-alphas({len(params.LEARNING_RATES)})-gammas({len(params.DISCOUNTS)})-{time.strftime('%Y%m%d-%H%M%S')}.csv", "w")
expected_SARSA_file = open(f"Files/expected_SARSA_stats-episodes({params.EPISODES})-alphas({len(params.LEARNING_RATES)})-gammas({len(params.DISCOUNTS)})-{time.strftime('%Y%m%d-%H%M%S')}.csv", "w")
#a_dict = {"a": 1, "b": 2}

writer = csv.writer(QLearning_file)
writer2 = csv.writer(SARSA_file)
writer3 = csv.writer(expected_SARSA_file)
for key, value in QLearning_stats.items():
    writer.writerow([key, value])
for key, value in SARSA_stats.items():
    writer2.writerow([key, value])    
for key, value in expectedSARSA_stats.items():
    writer3.writerow([key, value])

QLearning_file.close()
SARSA_file.close()
expected_SARSA_file.close()


#########################################################################



