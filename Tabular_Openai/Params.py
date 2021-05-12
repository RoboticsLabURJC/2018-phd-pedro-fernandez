import gym
import numpy as np



class Params():
    def __init__(self, env):

        # env
        self.env = env
        #self.observation_space.high = env.observation_space.high
        #self.num_actions = env.action_space.n
        #self.action_space = env.action_space


        # episodes
        self.EPISODES = 10_000

        # LEARNING_RATE = alpha = step_size. When alpha is close to 0, then we reward old values, not predcited values    
        self.LEARNING_RATE = 0.1 # alpha: min 0 - max 1
        self.LEARNING_RATES = np.linspace(0.1,1.0,10)

        # DISCOUNT rate = gamma. If gamma = 1, indicates future values have same value than current values
        self.DISCOUNT = 0.95 # gamma: min 0 - max 1
        self.DISCOUNTS = np.linspace(0.1,1.0,10)
        
        # for MountainCar
        self.REWARD_END = 0
        self.MAX_STEPS = 200

        # Epsilon
        self.EPSILON = 0.9  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL
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
        self.BEST_LEARNING_RATE = 1
        self.BEST_DISCOUNT = 1



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

        self.BEST_VALUE = -1000
        self.BEST_EPISODE = 0
        self.BEST_LEARNING_RATE = 1
        self.BEST_DISCOUNT = 1
        

    def set_bestValue(self, value):

        self.BEST_VALUE = value
        #print(f"bestReturn: {self.bestReturn}")         


    def get_bestValue(self):

        return self.BEST_VALUE
        #print(f"bestReturn: {self.bestReturn}")        

    def set_bestEpisode(self, episode):

        self.BEST_EPISODE = episode
        #print(f"bestReturn: {self.bestReturn}")         


    def get_bestEpisode(self):

        return self.BEST_EPISODE
        #print(f"bestReturn: {self.bestReturn}")  


    def set_bestLEARNING_RATE(self, learning):

        self.BEST_LEARNING_RATE = learning
        #print(f"bestReturn: {self.bestReturn}")         


    def get_bestLEARNING_RATE(self):

        return self.BEST_LEARNING_RATE
        #print(f"bestReturn: {self.bestReturn}")          


    def set_bestDISCOUNT(self, discount):

        self.BEST_DISCOUNT = discount
        #print(f"bestReturn: {self.bestReturn}")         


    def get_bestDISCOUNT(self):

        return self.BEST_DISCOUNT
        #print(f"bestReturn: {self.bestReturn}")          