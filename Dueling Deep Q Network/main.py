"""
Implementation of vanilla DQN using OpenAI LunarLander
"""
import gym
from agent import env, n_act, Agent
import matplotlib
import matplotlib.pyplot as plt
from IPython import display


    
#Define the needed hyperparameters for training.
batch = 128 #Number of transitions sampled inside the replay buffer
gamma = 0.99 #Discount rate
start_eps = 0.9 #Epsilon starting value
learn_rate = 1e-4 #Learning rate of the neural net

#(self, learn_rate, gamma, n_act, epsilon, batch)
Dueling_Agent = Agent(learn_rate, gamma, n_act, start_eps, batch)
Dueling_Agent.train()
