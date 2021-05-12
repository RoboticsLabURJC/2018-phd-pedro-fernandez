import numpy as np
 
class Agent:
    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """
    def choose_action(self, state):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action




class Params():
    def __init__(self, env):

        # env
        self.env = env
        #self.observation_space.high = env.observation_space.high
        #self.num_actions = env.action_space.n
        #self.action_space = env.action_space


        # episodes
        self.EPISODES = 10_000

        # LEARNING_RATE = alpha. When alpha is close to 0, then we reward old values, not predcited values    
        self.LEARNING_RATE = 0.1 # alpha: min 0 - max 1
        
        # DISCOUNT rate = gamma. If gamma = 1, indicates future values have same value than current values
        self.DISCOUNT = 0.95 # gamma: min 0 - max 1
        
        # for MountainCar
        self.REWARD_END = 0
        self.MAX_STEPS = 200

        # Epsilon
        self.EPSILON = 0.5  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = self.EPISODES #we can play with this var
        self.epsilon_decay_value = self.EPSILON/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)
        
        #Table
        self.DISCRETE_VALUE = 40
        self.DISCRETE_OS_SIZE = [self.DISCRETE_VALUE] * len(self.env.observation_space.high)
        self.discrete_os_win_size = (self.env.observation_space.high -
                        self.env.observation_space.low)/self.DISCRETE_OS_SIZE

        self.DISCRETE = True

        #for stats
        self.SHOW_EVERY = 200
        self.MIN_VALUE = -200 # -200 for mountainCar
        # Save Rewards in each episode

        self.STATS_EVERY = 200
        self.SAVE_TABLE_EVERY = 1000

        self.BEST_VALUE = -1000
        self.BEST_EPISODE = 0



    def set_epsilon(self, epsilon_decay_value):
        self.EPSILON -= epsilon_decay_value  

    def reset_epsilon(self):
        self.EPSILON = 0.5   


    def reset_Values(self):
        #self.LEARNING_RATE = 0.1 # alpha: min 0 - max 1
        
        # DISCOUNT rate = gamma. If gamma = 1, indicates future values have same value than current values
        #self.DISCOUNT = 0.95 # gamma: min 0 - max 1
        
        # for MountainCar
        self.REWARD_END = 0
        self.MAX_STEPS = 200

        # Epsilon
        self.EPSILON = 0.5  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = self.EPISODES #we can play with this var
        self.epsilon_decay_value = self.EPSILON/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)
        
        #Table
        self.DISCRETE_VALUE = 40
        self.DISCRETE_OS_SIZE = [self.DISCRETE_VALUE] * len(self.env.observation_space.high)
        self.discrete_os_win_size = (self.env.observation_space.high -
                        self.env.observation_space.low)/self.DISCRETE_OS_SIZE

        self.DISCRETE = True

        #for stats
        self.SHOW_EVERY = 200
        self.MIN_VALUE = -200 # -200 for mountainCar
        # Save Rewards in each episode

        self.STATS_EVERY = 200
        self.SAVE_TABLE_EVERY = 1000



 
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
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = len(num_state)
        self.num_actions = num_actions
 
        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space
 
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


class QLearningAgent(Agent):
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
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = len(num_state)
        self.num_actions = num_actions
 
        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space
    def update(self, state, state2, reward, action, action2):
        """
        Update the action value function using the Q-Learning update.
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
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] += self.alpha * (target - predict)



class ExpectedSarsaAgent(Agent):
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
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = len(num_state)
        self.num_actions = num_actions
 
        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the Expected SARSA update.
        Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
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
 
        expected_q = 0
        q_max = np.max(self.Q[next_state, :])
        greedy_actions = 0
        for i in range(self.num_actions):
            if self.Q[next_state][i] == q_max:
                greedy_actions += 1
     
        non_greedy_action_probability = self.epsilon / self.num_actions
        greedy_action_probability = ((1 - self.epsilon) / greedy_actions) + non_greedy_action_probability
 
        for i in range(self.num_actions):
            if self.Q[next_state][i] == q_max:
                expected_q += self.Q[next_state][i] * greedy_action_probability
            else:
                expected_q += self.Q[next_state][i] * non_greedy_action_probability
 
        target = reward + self.gamma * expected_q
        self.Q[prev_state, prev_action] += self.alpha * (target - predict)        