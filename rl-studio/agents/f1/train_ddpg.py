from collections import deque
import time
import random
import os
import time
from tqdm import tqdm
#from cprint import cprint
import numpy as np
import random
import utils
#from icecream import ic
from datetime import datetime, timedelta
import numpy as np
import gym
import pandas as pd
from algorithms.ddpg import ModifiedTensorBoard, OUActionNoise, Buffer, DDPGAgent
from envs.gazebo_env import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from visual.ascii.images import JDEROBOT_LOGO
from visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO

#ic.enable()
#ic.configureOutput(prefix=f'{datetime.now()} | ')


def save_stats_episodes(environment, outdir, aggr_ep_rewards, current_time):
    '''
            We save info of EPISODES in a dataframe to export or manage
    '''

    outdir_episode = f"{outdir}_stats"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file_csv = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.csv"
    file_excel = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.xlsx"

    df = pd.DataFrame(aggr_ep_rewards)
    df.to_csv(file_csv, mode='a', index = False, header=None)
    df.to_excel(file_excel)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_messages(*args, **kwargs):

    print(f"\n\t{bcolors.OKCYAN}====>\t{args[0]}:{bcolors.ENDC}\n")
    for key, value in kwargs.items():
        print(f"\t{bcolors.OKBLUE}[INFO] {key} = {value}{bcolors.ENDC}")
    print(f"\n")    


class F1TrainerDDPG:

    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # var to config Agents
        self.config = dict(params)

        ## vars to config function main ddpg
        self.agent_name = params.agent["name"]
        self.model_state_name = params.settings["model_state_name"]
        # environment params
        self.outdir = f"{params.settings['output_dir']}{params.algorithm['name']}_{params.agent['name']}_{params.environment['params']['sensor']}"
        self.ep_rewards = [] 
        self.aggr_ep_rewards = {
            'episode':[], 'avg':[], 'max':[], 'min':[], 'step':[], 'epoch_training_time':[], 'total_training_time':[]
        }
        self.best_current_epoch = {'best_epoch':[], 'highest_reward':[], 'best_step':[], 'best_epoch_training_time':[], 'current_total_training_time':[]}
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.total_episodes = params.settings["total_episodes"]
        self.training_time = params.settings["training_time"]
        self.save_episodes = params.settings["save_episodes"]
        self.save_every_step = params.settings["save_every_step"]
        self.estimated_steps = params.environment["params"]["estimated_steps"]
        
        # algorithm params
        self.tau = params.algorithm["params"]["tau"]
        self.gamma = params.algorithm["params"]["gamma"]
        self.std_dev = params.algorithm["params"]["std_dev"]
        self.model_name = params.algorithm["params"]["model_name"]
        self.buffer_capacity = params.algorithm["params"]["buffer_capacity"]
        self.batch_size = params.algorithm["params"]["batch_size"]
                
        # States
        self.state_space = params.agent["params"]["states"]["state_space"]
        self.states = params.agent["params"]["states"]
        #self.x_row = params.agent["params"]["states"][self.state_space][0]
   
        # Actions
        self.action_space = params.environment["actions_set"]
        self.actions = params.environment["actions"]
        self.actions_size = params.environment["actions_number"]
   
        # Rewards
        self.reward_function = params.agent["params"]["rewards"]["reward_function"]
        self.highest_reward = params.agent["params"]["rewards"][self.reward_function]["highest_reward"] 

        # Env
        self.environment = {}
        self.environment['agent'] = params.agent["name"]
        self.environment['model_state_name'] = params.settings["model_state_name"]
        # Env
        self.environment['env'] = params.environment["params"]["env_name"]
        self.environment['training_type'] = params.environment["params"]["training_type"]
        self.environment['circuit_name'] = params.environment["params"]["circuit_name"]
        self.environment['launch'] = params.environment["params"]["launch"]
        self.environment['gazebo_start_pose'] = [params.environment["params"]["circuit_positions_set"][1][0],params.environment["params"]["circuit_positions_set"][1][1]]
        self.environment['alternate_pose'] = params.environment["params"]["alternate_pose"]
        self.environment['gazebo_random_start_pose'] = params.environment["params"]["circuit_positions_set"]    
        self.environment['estimated_steps'] = params.environment["params"]["estimated_steps"]
        self.environment['sensor'] = params.environment["params"]["sensor"]
        self.environment['telemetry_mask'] = params.settings["telemetry_mask"]
        
        # Image
        self.environment['height_image'] = params.agent["params"]["camera_params"]["height"]
        self.environment['width_image'] = params.agent["params"]["camera_params"]["width"]
        self.environment['center_image'] = params.agent["params"]["camera_params"]["center_image"]
        self.environment['image_resizing'] = params.agent["params"]["camera_params"]["image_resizing"]
        self.environment['new_image_size'] = params.agent["params"]["camera_params"]["new_image_size"]
        self.environment['raw_image'] = params.agent["params"]["camera_params"]["raw_image"]
        self.environment['num_regions'] = params.agent["params"]["camera_params"]["num_regions"]

        # States
        self.environment['state_space'] = params.agent["params"]["states"]["state_space"]
        self.environment['states'] = params.agent["params"]["states"][self.state_space]
        self.environment['x_row'] = params.agent["params"]["states"][self.state_space][0]

        # Actions
        self.environment['action_space'] = params.environment["actions_set"]
        self.environment['actions'] = params.environment["actions"]
        self.environment['beta_1'] = -(params.environment["actions"]['w_left'] / params.environment["actions"]['v_max'])   
        self.environment['beta_0'] = -(self.environment['beta_1'] * params.environment["actions"]['v_max'])   

        # Rewards
        self.environment['reward_function'] = params.agent["params"]["rewards"]["reward_function"]
        self.environment['rewards'] = params.agent["params"]["rewards"][self.reward_function]
        self.environment['min_reward'] = params.agent["params"]["rewards"][self.reward_function]["min_reward"]

        # Algorithm
        self.environment['critic_lr'] = params.algorithm['params']['critic_lr']
        self.environment['actor_lr'] = params.algorithm['params']['actor_lr']
        self.environment['model_name'] = params.algorithm['params']['model_name']

        # 
        self.environment['ROS_MASTER_URI'] = params.settings["ros_master_uri"]
        self.environment['GAZEBO_MASTER_URI'] = params.settings["gazebo_master_uri"]
        self.environment['telemetry'] = params.settings["telemetry"]

        print(f"\t[INFO]: environment: {self.environment}\n")

        # Env
        self.env = gym.make(self.env_name, **self.environment)

    def __repr__(self):
        return print(f"\t[INFO]: self.config: {self.config}")

    def main(self):

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        print(JDEROBOT)
        print(JDEROBOT_LOGO)

        os.makedirs(f"{self.outdir}", exist_ok=True)

        start_time_training = time.time()
        telemetry_start_time = time.time()
        start_time = datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")
        best_epoch = 1
        current_max_reward = 0
        best_step = 0

        # Reset env
        state, state_size = self.env.reset()   

        # Checking state and actions 
        print_messages('In train_ddpg.py', state_size = state_size, action_space = self.action_space, action_size = self.actions_size)

        ## --------------------- Deep Nets ------------------
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        # Init Agents
        ac_agent = DDPGAgent(self.environment, self.actions_size, state_size, self.outdir)
        #init Buffer
        buffer = Buffer(state_size, self.actions_size, self.state_space, self.action_space, self.buffer_capacity, self.batch_size)
        #Init TensorBoard
        tensorboard = ModifiedTensorBoard(log_dir=f"{self.outdir}/logs_TensorBoard/{self.model_name}-{time.strftime('%Y%m%d-%H%M%S')}")

        ## -------------    START TRAINING -------------------- 
        print(LETS_GO)
        for episode in tqdm(range(1, self.total_episodes + 1), ascii=True, unit='episodes'):
            tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            prev_state, prev_state_size = self.env.reset()        

            # ------- WHILE
            #while not done and (step < self.estimated_steps) and (datetime.now() - timedelta(hours=self.training_time) < start_time):
            #while not done and (step < self.estimated_steps):
            while not done:

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0) 
                # Get action
                action = ac_agent.policy(tf_prev_state, ou_noise, self.action_space)
                #print_messages('action in every step', action=action, step=step)

                state, reward, done, info = self.env.step(action)
                cumulated_reward += reward    

                buffer.record((prev_state, action, reward, state))
                buffer.learn(ac_agent, self.gamma)
                ac_agent.update_target(ac_agent.target_actor.variables, ac_agent.actor_model.variables, self.tau)
                ac_agent.update_target(ac_agent.target_critic.variables, ac_agent.critic_model.variables, self.tau)

                prev_state = state
                step += 1

                # save best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = datetime.now() - start_time_epoch

                 # Showing stats in screen only for monitoring. Showing every 'save_every_step' value
                if not step % self.save_every_step:
                    print_messages('Showing stats but not saving...', current_episode = episode, current_step = step, cumulated_reward_in_this_episode = int(cumulated_reward), 
                        total_training_time = (datetime.now() - start_time), epoch_time = datetime.now() - start_time_epoch)
                    print_messages('... and best record', best_episode_until_now = best_epoch, 
                        in_best_step = best_step, with_highest_reward = int(current_max_reward), in_best_epoch_trining_time = best_epoch_training_time)

                # save at completed steps    
                if step >= self.estimated_steps:
                    done = True
                    print_messages('Lap completed in:', time = datetime.now() - start_time_epoch, in_episode = episode, episode_reward = int(cumulated_reward), with_steps = step)
                    ac_agent.actor_model.save(f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")
                    ac_agent.critic_model.save(f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")

            # Save best lap
            if cumulated_reward > self.highest_reward:
                self.highest_reward = cumulated_reward
                print_messages('Saving best lap', best_episode_until_now = best_epoch, in_best_step = best_step, with_highest_reward = int(cumulated_reward), 
                        in_best_epoch_trining_time = best_epoch_training_time, total_training_time = (datetime.now() - start_time))
                self.best_current_epoch['best_epoch'].append(best_epoch)
                self.best_current_epoch['highest_reward'].append(cumulated_reward)
                self.best_current_epoch['best_step'].append(best_step)
                self.best_current_epoch['best_epoch_training_time'].append(best_epoch_training_time)
                self.best_current_epoch['current_total_training_time'].append(datetime.now() - start_time)
                save_stats_episodes(self.environment, self.outdir, self.best_current_epoch, start_time)
                ac_agent.actor_model.save(f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")
                ac_agent.critic_model.save(f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")


            # ended at training time setting: 2 hours, 15 hours...
            if (datetime.now() - timedelta(hours=self.training_time) > start_time):
                print_messages('Training time finished in:', time = datetime.now() - start_time, episode = episode, cumulated_reward = cumulated_reward, total_time = (datetime.now() - timedelta(hours=self.training_time)))
                if self.highest_reward < cumulated_reward:
                    ac_agent.actor_model.save(f"{self.outdir}/models/{self.model_name}_TRAININGTIME_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")
                    ac_agent.critic_model.save(f"{self.outdir}/models/{self.model_name}_TRAININGTIME_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")

                break

            # WE SAVE BEST VALUES IN EVERY EPISODE
            self.ep_rewards.append(cumulated_reward)
            if not episode % self.save_episodes:    
                average_reward = sum(self.ep_rewards[-self.save_episodes:]) / len(self.ep_rewards[-self.save_episodes:])
                min_reward = min(self.ep_rewards[-self.save_episodes:])
                max_reward = max(self.ep_rewards[-self.save_episodes:])
                tensorboard.update_stats(reward_avg=average_reward, reward_max=max_reward, steps = step)

                print_messages('Showing batch:', current_episode_batch = episode, max_reward_in_current_batch = int(max_reward), best_epoch_in_all_training = best_epoch, highest_reward_in_all_training = int(max(self.ep_rewards)), in_best_step = best_step, total_time = (datetime.now() - start_time))
                self.aggr_ep_rewards['episode'].append(episode)
                self.aggr_ep_rewards['step'].append(step)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max_reward)
                self.aggr_ep_rewards['min'].append(min_reward)
                self.aggr_ep_rewards['epoch_training_time'].append((datetime.now()-start_time_epoch).total_seconds())
                self.aggr_ep_rewards['total_training_time'].append((datetime.now()-start_time).total_seconds())

                #if max_reward > max(self.ep_rewards):
                if max_reward > self.highest_reward:
                    print_messages('Saving batch', max_reward = int(max_reward))
                    tensorboard.update_stats(reward_avg=int(average_reward), reward_max=int(max_reward), steps = step)
                    ac_agent.actor_model.save(f"{self.outdir}/models/{self.model_name}_ACTOR_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")
                    ac_agent.critic_model.save(f"{self.outdir}/models/{self.model_name}_CRITIC_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model")
                    save_stats_episodes(self.environment, self.outdir, self.aggr_ep_rewards, start_time)

        save_stats_episodes(self.environment, self.outdir, self.aggr_ep_rewards, start_time)
        self.env.close()

