import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.optim as optim
import torch.nn as nn
from IPython import display

is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from IPython import display

from buffer import Transition, ReplayBuffer
from nets import Nets

env = gym.make('LunarLander-v2', render_mode='human', enable_wind = True)

#Check out the state space
n_obs = len(env.reset())
n_act = env.action_space.n
print('State shape: ', n_obs)#env.observation_space.shape)
print('Number of actions: ', n_act )

# By default, we use CPU. If CUDA is available, this command will automatically call it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    
    def __init__(self, learn_rate, gamma, n_act, epsilon, batch):
        #self.action_space = [i for i in range(n_act)]
        self.gamma = gamma #Discount rate
        self.epsilon = epsilon #episode starts
        self.batch_size = batch #Number of transitions sampled inside the replay buffer
        self.epsilon_decay = 1000 #Episode decay rate
        self.epsilon_final = 0.05 #Episode final rate
        self.update_rate = 0.005
        self.act = n_act
        self.step_counter = 0
        self.learn_rate = learn_rate
        state, _ = env.reset()
        no_obs = len(state)
        self.policy_net = Nets(no_obs, self.act).to(device)
        self.target_net = Nets(no_obs, self.act).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    memory = ReplayBuffer(10000)   
    steps_done = 0
    def select_policy(self, state):
        #global steps_done
        sample = random.random()
        eps_threshold = self.epsilon_final + (self.epsilon - self.epsilon_final) * \
            math.exp(-1. * self.step_counter / self.epsilon_decay)
        self.step_counter  += 1
        if sample > eps_threshold:
            with torch.no_grad():
            # Contrary to the vanilla DQN, in Double (Dueling) DQN only the maximum argument of the next state is taken into consideration.
                 return self.policy_net(state).argmax(1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        
        optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learn_rate, amsgrad=True)
    
        
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states)
    # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
        optimizer.zero_grad()
        loss.backward()
    # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer.step()
    
    episode_durations = []
    def plot_durations(self, show_result=False):
        
        #Showing the real-time plotting of the training.
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            env.close()
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
            plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    
    def train (self):
        if torch.cuda.is_available():
            num_episodes = 600
        else:
            num_episodes = 150
        #memory = ReplayBuffer(10000)
        for i_episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            #env.render()
            
            for t in count():
                action = self.select_policy(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                
                # Move to the next state
                state = next_state
                #optimize your model
                self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.update_rate + target_net_state_dict[key]*(1-self.update_rate)
                    self.target_net.load_state_dict(target_net_state_dict)
                    
                if done:
                    #self.episode_durations.append(t + 1)
                    #self.plot_durations()
                    break
                
        print('Complete')
        #self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
        
    
