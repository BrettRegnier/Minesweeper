
import random
import torch
import torch.nn as nn
import numpy as np
import collections

class BroomDQL(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BroomDQL, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 80, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, stride=1),
            nn.ReLU()
        )

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))
        # print(conv_out_shape); exit()
        self._fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._fc(conv_out)

Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'next_state'])
class Agent:
    def __init__(self, env, memory):
        self._env = env
        self._memory = memory
        self.Reset()

    def Reset(self, soft=False):
        self._state = self._env.reset(soft=soft)
        self._total_reward = 0

    @torch.no_grad()
    def PlayStep(self, net, steps, epsilon=0.0, device="cpu"):
        done_reward = None
        win = False

        # choose random action for exploration
        if np.random.random() < epsilon:
            action = self._env.action_space.sample()
        else:
            current_state = np.array([self._state], copy=False)
            state_value = torch.FloatTensor(current_state).to(device)
            q_value = net(state_value)
            value, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())

        #step
        next_state, reward, done, win = self._env.step(action)
            

        # self._env.render()
        self._total_reward += reward

        experience = Experience(self._state, action, reward, done, next_state)
        if not (steps == 0 and done):
            self._memory.append(experience)
        self._state = next_state

        if done:
            done_reward = self._total_reward
            self.Reset()
        return done_reward, win
    
class AlternateAgent:
    def __init__(self, env, memory):
        self._env = env
        self._memory = memory
        self.Reset()
    
    def Reset(self):
        self._total_reward = 0
    
    @torch.no_grad()
    def PlayStep(self, net, lg_state, steps, epsilon=0.0, device='cpu'):
        done_reward = None
        win = False
        
        if np.random.random() < epsilon:
            action = self._env.action_space.sample()
        else:
            current_state = np.array([lg_state], copy=False)
            state_value = torch.FloatTensor(current_state).to(device)
            q_value = net(state_value)
            value, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())
            
        next_state, reward, done, win = self._env.step(action)
        
        self._total_reward += reward
        
        if done:
            done_reward = self._total_reward
        
        return next_state, action, reward, done, done_reward, win
        
    def AddMemory(self, state, action, reward, done, next_state):
        experience = Experience(state, action, reward, done, next_state)
        self._memory.append(experience)      
    
class Memory:
    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def append(self, experience):
        self._buffer.append(experience)

    def sample(self, batch_size):        
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for i in indices:
            states.append(self._buffer[i].state)
            actions.append(self._buffer[i].action)
            rewards.append(self._buffer[i].reward)
            dones.append(self._buffer[i].done)
            next_states.append(self._buffer[i].next_state)

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.int8), np.array(next_states)
