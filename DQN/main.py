import numpy as np
#import keras.backend.tensorflow_backend as backend
#import tf.keras.backend.tensorflow_backend.set_session() -> tf.compat.v1.keras.backend.set_session() as backend

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
#from collections import deque
#import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

from Params import Params
from DeepAgents import ModifiedTensorBoard, DQNAgent
from environments import BlobEnv, Blob
import gym


# Using the gym library to create the environment
#env = gym.make('MountainCar-v0')
env = BlobEnv()
#env.reset()

params = Params(env)

#params.env.reset()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)
tf.compat.v1.random.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


agent = DQNAgent(params)


#--------------------------------------------------
# Iterate over episodes
for episode in tqdm(range(1, params.EPISODES + 1), ascii=True, unit='episodes'):

    epsilon = params.EPSILON
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if params.SHOW_PREVIEW and not episode % params.AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % params.AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-params.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-params.AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-params.AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-params.AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= params.MIN_REWARD:
            agent.model.save(f'models/{params.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > params.MIN_EPSILON:
        epsilon *= params.EPSILON_DECAY
        epsilon = max(params.MIN_EPSILON, epsilon)