
from keras.models import Sequential  
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.python.keras.engine.training import _minimum_control_deps
#import tensorflow.keras.backend as K
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras import Model

from collections import deque 
import numpy as np
import gym 
import time
import random
from tqdm import tqdm
import os


# Own Tensorboard class. Be careful, this works for MAC
class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()



class DQNAgent_noImage:
    def __init__(self, sess, action_dim, observation_dim):
        # Force keras to use the session that we have created
        K.set_session(sess)
        #tf.compat.v1.keras.backend.set_session(sess)

        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.model = self.create_model()


        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self):
        state_input = Input(shape=(self.observation_dim))
        state_h1 = Dense(400, activation='relu')(state_input)
        state_h2 = Dense(300, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='linear')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(0.005))
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)    


    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0





DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MINIMUM_REPLAY_MEMORY = 1_000
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False
SHOW_EVERY = 200


ENV_NAME = 'MountainCar-v0'

# Environment details
env = gym.make(ENV_NAME)
action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# creating own session to use across all the Keras/Tensorflow models we are using
sess = tf.compat.v1.Session()

# Replay memory to store experiances of the model with the environment
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# Our models to solve the mountaincar problem.
agent = DQNAgent_noImage(sess, action_dim, observation_dim)


#tf.set_random_seed(2212)




def train_dqn_agent():
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
    X_cur_states = []
    X_next_states = []
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        X_cur_states.append(cur_state)
        X_next_states.append(next_state)
    
    X_cur_states = np.array(X_cur_states)
    X_next_states = np.array(X_next_states)
    
    # action values for the current_states
    cur_action_values = agent.model.predict(X_cur_states)
    # action values for the next_states taken from our agent (Q network)
    next_action_values = agent.model.predict(X_next_states)
    for index, sample in enumerate(minibatch):
        cur_state, action, reward, next_state, done = sample
        if not done:
            # Q(st, at) = rt + DISCOUNT * max(Q(s(t+1), a(t+1)))
            cur_action_values[index][action] = reward + DISCOUNT * np.amax(next_action_values[index])
        else:
            # Q(st, at) = rt
            cur_action_values[index][action] = reward
    # train the agent with new Q values for the states and the actions
    agent.model.fit(X_cur_states, cur_action_values, verbose=0)



# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.compat.v1.random.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')



for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    episode_reward = 0
    episode_length = 0

    if episode % SHOW_EVERY == 0:
        print("episode:", episode)
        render = True
    else:
        render = False

    while not done: 

        episode_length += 1

        if render:
            env.render()   


        # ACTION
        if np.random.rand() <= epsilon:
            #action = np.random.randint(0, env.action_space.n, size=1)[0]
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(agent.model.predict(np.expand_dims(current_state, axis=0))[0])


        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))

        # train with method...
        agent.train(done, step)

        #...or We can use a function defined below 
        #if(len(replay_memory) < MINIMUM_REPLAY_MEMORY):
        #    continue
        #train_dqn_agent()

        ####
        current_state = new_state
        step += 1

   # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)   


    # some stats
    #max_reward = max(episode_reward, max_reward)
    print('Episode', episode, 'Episodic Reward', episode_reward, 'EPSILON', epsilon)    