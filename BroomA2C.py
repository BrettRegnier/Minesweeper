import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np

class BroomConvoA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BroomConvoA2C, self).__init__()

        self._conv = nn.Sequential(
            nn.Conv2d(1, 135, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        )

        out = conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))

        self._policy = nn.Sequential(
            nn.Linear(in_features=conv_out_shape, out_features=406),
            nn.Linear(in_features=406, out_features=1008),
            nn.Linear(in_features=1008, out_features=491),
            nn.Linear(in_features=491, out_features=395),
            nn.Linear(in_features=395, out_features=n_actions)
        )

        self._value = nn.Sequential(
            nn.Linear(in_features=conv_out_shape, out_features=406),
            nn.Linear(in_features=406, out_features=1008),
            nn.Linear(in_features=1008, out_features=491),
            nn.Linear(in_features=491, out_features=395),
            nn.Linear(in_features=395, out_features=1)
        )

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._policy(conv_out), self._value(conv_out)

class BroomA2C(nn.Module):
    def __init__(self, input_shape, n_actions, lr=0.0001):
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
        self._fc = nn.Sequential(
                nn.Linear(in_features=in_shape, out_features=406),
                nn.ReLU(),
                nn.Linear(in_features=406, out_features=395),
                nn.ReLU(),
                nn.Linear(in_features=395, out_features=n_actions)
            )


        self._optimizer = optim.Adam(self.parameters(), lr=self._lr)

    def forward(self, X):

        return self._fc(X)

class AgentA2C(object):
    def __init__(self, actor, critic):
        self._actor = actor
        self._critic = critic

    def PlayStep(self, state, device="cpu"):
        action = self.Act(state)
        next_state, reward, done, win = env.step(action)

        return log_probs, action, next_state, reward, done, win

    def Learn(self, state, reward, next_state, done, gamma=0.99):
        self._actor._optimizer.zero_grad()
        self._critic._optimizer.zero_grad()
        
        state_value, next_state_value = self.Critique(state, next_state)

        delta = ((reward + gamma * next_state_value * (1-int(done))) - state_value)

        actor_loss = -self._log_probs * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self._actor._optimizer.step()
        self._critic._optimizer.step()
    
    def Act(self, state):      
        probabilities = F.softmax(self._actor(state))  
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self._log_probs = action_probs.log_prob(action)

        return action.item()

    def Critique(self, state, next_state):
        return self._critic(state), self._critic(next_state)