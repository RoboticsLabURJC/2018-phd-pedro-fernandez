
 
import numpy as np
from Agent import Agent
from Params import Params
 
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

        #if not discrete_size_table:
        #    self.discrete_size_table = discrete_size_table
        #print(f"self.discrete_os_size: {self.discrete_os_size}, self.num_state: {self.num_state},\
    #self.num_actions: {self.num_actions}, self.action_space: {self.action_space}, \
        #self.discrete: {self.discrete}")

        #self.Q = np.zeros((self.num_state, self.num_actions))
        #if not self.discrete:
        #    print(f"discrete_size_table: {self.discrete}")
        #    self.Q_table = np.zeros((self.num_state, self.num_actions))
        #else:
        #    self.Q_table = np.random.uniform(low=-2, high=0, size=([self.discrete_os_size] * self.num_state + [self.num_actions]))
        
        #self.Q_table = np.random.uniform(low=-2, high=0, size=([self.discrete_os_size] * self.num_state + [self.num_actions]))


    def choose_action_inside(self, state):

        #action = 0
        #if np.random.uniform(0, 1) < self.epsilon:
        #    action = self.action_space.sample()
        #else:
        #    action = np.argmax(self.Q[state, :])

        if np.random.random() > self.epsilon: #IF EPSILON = 1, then always RANDOM actions (EXPLORATORY)
            # Get action from Q table
            action = np.argmax(self.Q_table[state])
        else:
            # Get random action
            #action = np.random.randint(0, self.action_space.n)
            action = np.random.randint(0, self.num_actions)


        return action



    def update(self, state, state2, reward, action):
        """
        Update the action value function using the Q-Learning update.
        (1) Q(S, A) = Q(S, A) + alpha(reward + (gamma * maxQ(S_, A_) - Q(S, A))

        or 
        (2) Q(S, A) = (1 - alpha) * Q(S, A) + alpha(reward + gamma * maxQ(S_, A))

        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        #predict = self.Q[state, action]
        #target = reward + self.gamma * np.max(self.Q[state2, :])
        #self.Q[state, action] += self.alpha * (target - predict)
        #self.Q_table[state + (action.)] = (1 - self.learningrate) * self.Q_table[state, action] \
        #    + self.learningrate * (reward + self.discountrate * np.max(self.Q_table[state2, :]))


        #self.max_future_q = np.max(self.Q_table[state2])
        #self.current_q = self.Q_table[state + (action,)]

        #self.Q_table[state + (action,)] = (1 - self.learningrate) * self.current_q \
        #    + self.learningrate * (reward + self.discountrate * self.max_future_q)

        self.Q_table[state + (action,)] = (1 - self.learningrate) * self.Q_table[state + (action,)] + self.learningrate * \
                        (reward + self.discountrate * np.max(self.Q_table[state2]))



    #def draw_rewards2(self, aggr_ep_rewards, LEARNING_RATE, DISCOUNT):

    #    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    #    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    #    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
    #    plt.title('Q_Learning with Learning Rate and Discount Rate')
    #    plt.suptitle([LEARNING_RATE, DISCOUNT])
    #    plt.legend(loc=0) #loc=0 best place
    #    plt.grid(True)
        #plt.draw()
    #    os.makedirs("Images", exist_ok = True)            
    #    plt.savefig(f"Images/{type(self).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-{time.strftime('%Y%m%d-%H%M%S')}.jpg", bbox_inches='tight')
        #plt.show() 


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
