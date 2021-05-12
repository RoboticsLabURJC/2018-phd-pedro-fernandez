'''
https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/

'''


import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor
from time import sleep


env = gym.make("MountainCar-v0")
env.reset()

"""
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
"""

#-------------------------------
#Hyperparams
EPISODES = 25_000
LEARNING_RATE = 0.1 # min 0 - max 1
DISCOUNT = 0.95 # min 0 - max 1

REWARD_END = 0
#-------------------------------
# Exploration settings
epsilon = 1  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL
START_EPSILON_DECAYING = 1
#END_EPSILON_DECAYING = EPISODES//2 #we can play with this var
END_EPSILON_DECAYING = EPISODES/100 #we can play with this var
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)



#-------------------------------
# we have to discretize continuum variables. We have to find the optimal Qtable size for the problem
#DISCRETE_OS_SIZE = [20, 20]
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE


# create Q=Table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#-------------------------------
#for stats
SHOW_EVERY = 200
#print(EPISODES//2)
# Save Rewards in each episode
# For stats
STATS_EVERY = 100
SAVE_TABLE_EVERY = 1000
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


# print(discrete_os_win_size)
# print(q_table.shape)

#------------------------------
#to convert continuos states into discrete states. So we can index QTable with discrete_state return value: q_table[discrete_state] to get the values in those positions
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


#-------------------------------

# fix hyperparams
learning_rates_ = [0.1]
#learning_rates_ = [0.2, 0.5, 1]
#learning_ = np.zeros((len(learning_rates_), EPISODES))

#discount_rates_ = [0.95, 0.80, 0.5]
discount_rates_ = [0.95]
#discounts_ = np.zeros((len(discount_rates_), EPISODES))

#QTables_sizes_ = [20, 40, 80]
#q_tables_ = np.random.uniform(low=-2, high=0, size=([i for i in QTables_sizes_] + [env.action_space.n]))

#algorithms_ = ['QLearning', 'SARSA', 'Dyna+', 'DQL']

#-------------------------------
# draw 3D map
# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

#def print_map(value, episode, axes):
def print_3D(episode):
    grid_size = 40
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)
    # positionStep = (POSITION_MAX - POSITION_MIN) / grid_size
    # positions = np.arange(POSITION_MIN, POSITION_MAX + positionStep, positionStep)
    # velocityStep = (VELOCITY_MAX - VELOCITY_MIN) / grid_size
    # velocities = np.arange(VELOCITY_MIN, VELOCITY_MAX + velocityStep, velocityStep)
    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
    axis_x = []
    axis_y = []
    
    fig = plt.figure(figsize=(40, 10))
    axes = fig.add_subplot(1, 1, 1, projection='3d')
    #axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            #axis_z.append(value)
            #print('value_function.cost_to_go(position, velocity):',value_function.cost_to_go(position, velocity))

    #ax.scatter(axis_x, axis_y, axis_z)
    axes.scatter(axis_x, axis_y)
    axes.set_xlabel('Position')
    axes.set_ylabel('Velocity')
    #ax.set_zlabel('Cost to go')
    axes.set_title('Episode %d' % (episode + 1))
    plt.show()


for i, learningrates in enumerate(learning_rates_):

    for j, discountrates in enumerate(discount_rates_):
        
        epsilon = 1  # not a constant, going to be decayed. When close 1, more likely perform EXPLORATORY (random actions) and MORE time getting GOAL

#-------------------------------
    # starting the Party!!
        for episode in range(EPISODES):

            #print(f"i: {i} y learningrates:{learningrates}")
            episode_reward = 0


            if episode % SHOW_EVERY == 0:
                print("episode:", episode)
                render = True
            else:
                render = False    

            discrete_state = get_discrete_state(env.reset())
            #print(env.reset())
            #print(f"discrete_state: {discrete_state} in episode: {episode}")   

            done = False
            while not done:

                #action = np.argmax(q_table[discrete_state])
                
                if np.random.random() > epsilon: #IF EPSILON = 1, then always RANDOM actions (EXPLORATORY)
                    # Get action from Q table
                    action = np.argmax(q_table[discrete_state])
                else:
                    # Get random action
                    action = np.random.randint(0, env.action_space.n)


                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                new_discrete_state = get_discrete_state(new_state)

                #env.render() #si dejo esta funciona la visualziacion
                #if render: #no esta funcionando bien del todo la visualazion
                #    env.render()
            #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # If simulation did not end yet after last step - update Q table
                if not done:

                # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(q_table[new_discrete_state])
                #print("np.max(q_table[new_discrete_state]", np.max(q_table[new_discrete_state]))
                    #print("q_table[new_discrete_state]:", q_table[new_discrete_state])
                #print("new_discrete_state:", new_discrete_state)

                # Current Q value (for current state and performed action)
                    current_q = q_table[discrete_state + (action,)]
                    #print("discrete_state:", discrete_state)
                    #print("action:", action)
                    #print("discrete_state + (action,):", discrete_state + (action,))
                    #print("current_q", current_q)

                # And here's our equation for a new Q value for current state and action
                    new_q = (1 - learningrates) * current_q + learningrates * \
                        (reward + discountrates * max_future_q)
                    #print("new_q:", new_q)

                # Update Q table with new Q value
                    q_table[discrete_state + (action,)] = new_q
                    #print(f"discrete_state: {discrete_state} + action: {action} + ")

            # Simulation ended (for any reason) - if goal position is achived - update Q value with reward directly
                elif new_state[0] >= env.goal_position:
                    print(f"finish in episode {episode}")
                    #q_table[discrete_state + (action,)] = reward
                    #print(f"discrete_state (initial): {discrete_state} and q_table[discrete_state + (action,)]: {q_table[discrete_state + (action,)]}, and new_discrete_state: {new_discrete_state}")
                    print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}, discount: {discountrates:>2.2f},Learning Rate: {learningrates}')

                    q_table[discrete_state + (action,)] = REWARD_END #0
                    #if episode > 600:
                    #    env.render()

                
                discrete_state = new_discrete_state


            # Decaying is being done every episode if episode number is within decaying range
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                #print(f'Entramos >----- current epsilon: {epsilon:>1.3f},epsilon_decay_value: {epsilon_decay_value:>1.5f}')
                epsilon -= epsilon_decay_value
                #print(f'current epsilon: {epsilon:>1.3f},epsilon_decay_value: {epsilon_decay_value:>1.5f}')

            # show some STATS 
            ep_rewards.append(episode_reward)

            if not episode % STATS_EVERY:
                #average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
                average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
                aggr_ep_rewards['ep'].append(episode)
                aggr_ep_rewards['avg'].append(average_reward)
                aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
                aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
                print(f'Learning Rate: {learningrates}, discount: {discountrates:>2.2f}, episode: {episode:>5d}, average reward: {average_reward:>4.1f}\
,min reward: {min(ep_rewards[-STATS_EVERY:])}, max reward: {max(ep_rewards[-STATS_EVERY:])}\
,current epsilon: {epsilon:>1.3f},epsilon_decay_value: {epsilon_decay_value:>1.5f}')

            # lets save results in a table
            if episode % SAVE_TABLE_EVERY == 0 and episode > 0:
                np.save(f"qtables/{learningrates}-{discountrates}-{episode}-qtable.npy", q_table)

        #imagen en 3D
        #print_3D(episode)
        #plt.close()
        #fig = plt.figure()
        #while True:
        #plt.figure(1)
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
        plt.title('Q_Learning with Learning Rate and Discount Rate')
        plt.suptitle([learningrates, discountrates])
        plt.legend(loc=0) #loc=0 best place
        plt.grid(True)
                #plt.draw()
        plt.show()
        sleep(3)
        plt.clf()
        #plt.close('all')
        #sleep(1)

        #plt.close(fig)



env.close()  



# Plot Stats
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
#plt.legend(loc=1)
#plt.grid(True)
#plt.show()