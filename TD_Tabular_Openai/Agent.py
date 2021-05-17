# https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/
 

import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
import time
import os
 
class Agent:
    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """

    def choose_action(self, state):

        action = 0
        #if np.random.uniform(0, 1) < self.epsilon:
        #    action = self.action_space.sample()
        #else:
        #    action = np.argmax(self.Q[state, :])

        if np.random.random() > self.epsilon: #IF EPSILON = 1, then always RANDOM actions (EXPLORATORY)
            # Get action from Q table
            action = np.argmax(self.Q_table[state])
        else:
            # Get random action
            action = np.random.randint(0, self.action_space.n)

        return action



    def draw_rewards(self, aggr_ep_rewards, LEARNING_RATE, DISCOUNT, episode, MIN_VALUE):

        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
        plt.title(f"{type(self).__name__} with Learning Rate {LEARNING_RATE} and Discount Rate {DISCOUNT}")
        plt.suptitle([LEARNING_RATE, DISCOUNT])
        plt.legend(loc=0) #loc=0 best place
        plt.grid(True)
        os.makedirs("Images", exist_ok = True)            
        plt.savefig(f"Images/{type(self).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-({MIN_VALUE})-{time.strftime('%Y%m%d-%H%M%S')}.jpg", bbox_inches='tight')
        plt.clf()


    def tables_rewards(self, LEARNING_RATE, DISCOUNT, episode, q_table):

        os.makedirs("Tables", exist_ok = True)
                #np.save(f"Tables/{type(agent).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", q_table)
        np.save(f"Tables/{type(self).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", q_table)
       
    
    #def draw_2Variables(self, agents, all_rewards, episodes):
    #    for agent in agents:
    #        algorithm_means = np.array([np.mean(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
    #        algorithm_stds = np.array([sem(all_reward_sums[(agent, step_size)]) for step_size in step_sizes])
    #        plt.plot(step_sizes, algorithm_means, marker='o', linestyle='solid', label=type(agent).__name__)
    #        plt.fill_between(step_sizes, algorithm_means + algorithm_stds, algorithm_means - algorithm_stds, alpha=0.2)

    #    plt.legend(loc=0)
    #    plt.xlabel("Step-size")
    #    plt.ylabel("Sum of\n rewards\n per episode",rotation=0, labelpad=50)
    #    plt.xticks(step_sizes)
       
    #    os.makedirs("Images", exist_ok = True)            
    #    plt.savefig(f"Images/allRewards-{episodes}-{time.strftime('%Y%m%d-%H%M%S')}.jpg", bbox_inches='tight')
    #    plt.clf()
        #plt.show()