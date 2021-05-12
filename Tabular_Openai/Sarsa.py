 
import numpy as np
from Agent import Agent
from Params import Params

 
class SarsaAgent(Agent):
    """
    The Agent that uses SARSA update to improve it's behaviour
    """
    def __init__(self, epsilon, alpha, gamma, size_table, num_state, num_actions, action_space, discrete=True):
        """
        Constructor
        Args:

            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the random action
        """
        #self.Q_table = q_table
        self.epsilon = epsilon
        self.learningrate = alpha
        self.discountrate = gamma
        self.discrete_os_size = size_table
        self.num_state = len(num_state)
        self.num_actions = num_actions
        self.action_space = action_space
        self.discrete = discrete

#        self.bestValue = -1000
#        self.bestEpisode = 0
 
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the SARSA update.
        Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        predict = self.Q[prev_state, prev_action]
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)


#    def set_bestValue(self, value):

#        self.bestValue = value
        #print(f"bestReturn: {self.bestReturn}")         


#    def get_bestValue(self):

#        return self.bestValue
        #print(f"bestReturn: {self.bestReturn}")        

#    def set_bestEpisode(self, episode):

#        self.bestEpisode = episode
        #print(f"bestReturn: {self.bestReturn}")         


#    def get_bestEpisode(self):

#        return self.bestEpisode
        #print(f"bestReturn: {self.bestReturn}")             