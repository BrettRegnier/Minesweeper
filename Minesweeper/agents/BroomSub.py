
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections

from Replay import Experience

class Network(nn.Module):
    def __init__(self, input_shape, n_actions, lr):
        super(Network, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 80, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))

        self._fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self._optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._fc(conv_out)

class Agent:
    def __init__(self, input_shape, n_actions, lr, device):
        self._device = device
        self._n_actions = n_actions
        self._dql_net = Network((1, 3, 3), 2, lr).to(self._device)
        self._dql_target_net = Network((1, 3, 3), 2, lr).to(self._device)
        self._loss_func = nn.MSELoss()

    @torch.no_grad()
    def Act(self, state):
        state_np = np.array(state)
        q_states = []
        q_indices = []
        for i, tile_np in enumerate(state_np):
            # might need to add a surrounding []
            state_t = torch.tensor([[tile_np]]).to(self._device)
            q_value = self._dql_net(state_t)
            value, action_value = torch.max(q_value, dim=1)
            if action_value == 0:
                q_states.append(value)
                q_indices.append(i)
        
        if len(q_states) > 0:
            idx = np.argmax(q_states)
            action = q_indices[idx]
        else:
            action = random.randint(0, self._n_actions-1)

        return action

    # def Learn(self, state, )

    def LearnBatch(self, batch, gamma):
        self._dql_net._optimizer.zero_grad()

        # calculate loss
        states, actions, rewards, dones, next_states = batch

        states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(self._device)
        next_states_t = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64).to(self._device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.bool).to(self._device)

        preds = torch.tensor(np.zeros(len(states)) + 1, dtype=torch.int64).unsqueeze(-1).to(self._device)
        state_action_values = self._dql_net(states_t).gather(1, preds)
        # state_action_values = state_action_values
        state_action_values.squeeze_(-1)
        # state_action_values.gather(squeeze_(-1)
        with torch.no_grad():
            next_state_values = self._dql_target_net(next_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_t

        loss = self._loss_func(state_action_values, expected_state_action_values)
        loss.backward()
        self._dql_net._optimizer.step()

    def UpdateTarget(self):
        self._dql_target_net.state_dict(self._dql_net.state_dict())      
