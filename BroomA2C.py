import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import collections

import numpy as np

class BroomConvoA2C(nn.Module):
    def __init__(self, input_shape, n_actions, lr):
        super(BroomConvoA2C, self).__init__()

        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        )

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))

        self._fc1 = nn.Linear(in_features=conv_out_shape, out_features=1024)
        self._fc2 = nn.Linear(in_features=1024, out_features=512)
        self._policy = nn.Linear(in_features= 512, out_features=n_actions)
        self._value = nn.Linear(in_features=512, out_features=1)

        self._optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)

        out = F.relu(self._fc1(conv_out))
        out = F.relu(self._fc2(out))

        policy = self._policy(out)
        value = self._value(out)

        return policy, value

class BroomA2C(nn.Module):
    def __init__(self, input_shape, n_actions, lr=0.00001):
        super(BroomA2C, self).__init__()

        self._lr = lr

        in_shape = input_shape[1] * input_shape[2]

        # self._fc = nn.Sequential(
        #         nn.Linear(in_features=in_shape, out_features=406),
        #         nn.Linear(in_features=406, out_features=1008),
        #         nn.Linear(in_features=1008, out_features=491),
        #         nn.Linear(in_features=491, out_features=395),
        #         nn.Linear(in_features=395, out_features=n_actions)
        #     )

        self._fc1 = nn.Linear(in_features=in_shape, out_features=2048)
        self._fc2 = nn.Linear(in_features=2048, out_features=1536)
        self._policy = nn.Linear(in_features= 1536, out_features=n_actions)
        self._value = nn.Linear(in_features=1536, out_features=1)

        self._optimizer = optim.Adam(self.parameters(), lr=self._lr)

    def forward(self, X):
        out = F.relu(self._fc1(X))
        out = F.relu(self._fc2(out))

        policy = self._policy(out)
        value = self._value(out)

        return policy, value

class AgentA2C(object):
    def __init__(self, actor_critic):
        self._actor_critic = actor_critic

    def Learn(self, state, reward, next_state, done, gamma=0.99):
        self._actor_critic._optimizer.zero_grad()
        
        state_value, next_state_value = self.Critique(state, next_state)

        delta = reward + gamma * next_state_value
        temporal_difference = delta - state_value

        actor_loss = -self._log_probs * temporal_difference
        critic_loss = temporal_difference ** 2

        (actor_loss + critic_loss).backward()

        self._actor_critic._optimizer.step()
    
    def Act(self, state):
        probabilities, _ = self._actor_critic(state)  
        probabilities = F.softmax(probabilities, dim=0)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self._log_probs = action_probs.log_prob(action)

        return action.item()

    def Critique(self, state, next_state):
        _, state_value = self._actor_critic(state)
        _, next_state_value = self._actor_critic(next_state)
        return state_value, next_state_value

class Memory():
    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def append(self, exp):
        experience = Experience(exp[0], exp[1], exp[2], exp[3], exp[4])
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

    