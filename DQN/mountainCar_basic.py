from keras.models import Sequential  
from keras.layers import Dense, Flatten
from collections import deque 
import numpy as np
import gym 
env = gym.make('MountainCar-v0')  
import random


model = Sequential()
model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


D = deque()
observetime = 500
epsilon = 0.7
gamma = 0.9
mb_size = 50  


observation = env.reset() 
obs = np.expand_dims(observation, axis=0) 
state = np.stack((obs, obs), axis=1)
done = False
for t in range(observetime):
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        Q = model.predict(state)
        action = np.argmax(Q)
    observation_new, reward, done, info = env.step(action)
    obs_new = np.expand_dims(observation_new, axis=0)          
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     
    D.append((state, action, reward, state_new, done))         
    state = state_new         
    if done:
        env.reset()           
        obs = np.expand_dims(observation, axis=0)      
        state = np.stack((obs, obs), axis=1)
print('Observing Finished')


minibatch = random.sample(D, mb_size)                              
inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, env.action_space.n))

for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)
    
    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

    model.train_on_batch(inputs, targets)
print('Learning Finished')



observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()
    Q = model.predict(state)        
    action = np.argmax(Q)         
    observation, reward, done, info = env.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))