
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

from Params import Params
from environments import BlobEnv, Blob
from collections import deque
import time
import random
import os


# Own Tensorboard class
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


class DQNAgent:
    def __init__(self, params):

        self.ACTION_SPACE_SIZE = params.ACTION_SPACE_SIZE
        self.OBSERVATION_SPACE_VALUES = params.OBSERVATION_SPACE_VALUES
        self.OBSERVATION_SPACE_SHAPE = params.OBSERVATION_SPACE_SHAPE

        # main model  # gets trained every step
        #self.model = self.create_model()
        self.model = self.create_model_noImage()


        # DQN settings
        self.REPLAY_MEMORY_SIZE = params.REPLAY_MEMORY_SIZE  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = params.MIN_REPLAY_MEMORY_SIZE  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = params.MINIBATCH_SIZE  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = params.UPDATE_TARGET_EVERY  # Terminal states (end of episodes)
        self.MODEL_NAME = params.MODEL_NAME

        self.DISCOUNT = params.DISCOUNT # gamma: min 0 - max 1

        # Target model this is what we .predict against every step
        self.target_model = self.create_model_noImage()
        #self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


        # To set states and actions environment variables



        #self.env = env

    def create_model_noImage(self):
        model = Sequential()
        model.add(Dense(20, input_shape=(2,) + self.OBSERVATION_SPACE_SHAPE, activation='relu'))
        model.add(Flatten())       # Flatten input so as to have no problems with processing
        model.add(Dense(18, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model


    def create_model(self):
        model = Sequential()

        #model.add(Conv2D(256, (3, 3), input_shape=(2,) + self.OBSERVATION_SPACE_SHAPE))
        #model.add(Conv2D(256, (3, 3), input_shape=self.OBSERVATION_SPACE_VALUES))
        model.add(Conv2D(256, (3, 3), input_shape=(10, 10, 3)))

        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear"))
        #model.add(Dense(9, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


