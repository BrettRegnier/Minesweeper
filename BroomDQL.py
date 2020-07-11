
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections

from Memory import Experience

class Network(nn.Module):
    def __init__(self, input_shape, n_actions, lr):
        super(Network, self).__init__()        
        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 80, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, stride=1),
            nn.ReLU()
        )

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))

        self._fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self._optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._fc(conv_out)

class Agent:
    def __init__(self, input_shape, n_actions, sync_target, lr, device):
        self._dql_net = Network(input_shape, n_actions, lr).to(device)
        self._dql_target_net = Network(input_shape, n_actions, lr).to(device)
        self._loss_func = nn.MSELoss()
        self._sync_target = sync_target
        self._steps = 0

    @torch.no_grad()
    def PlayStep(self, net, steps, epsilon, device):
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

    @torch.no_grad()
    def Act(self, state, device):
        state_np = np.array([state], copy=False)
        state_t = torch.tensor(state_np).to(device)

        q_value = self._dql_net(state_t)
        value, action_value = torch.max(q_value, dim=1)
        action = int(action_value.item())

        self._steps += 1
    
        return action

    def LearnBatch(self, batch, gamma, device):
        # sync target
        if self._steps % self._sync_target == 0:
            self._dql_target_net.state_dict(self._dql_net.state_dict())        

        self._dql_net._optimizer.zero_grad()

        # calculate loss
        states, actions, rewards, dones, next_states = batch

        states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(device)
        next_states_t = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.bool).to(device)

        state_action_values = self._dql_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self._dql_target_net(next_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma * rewards_t

        loss = self._loss_func(state_action_values, expected_state_action_values)
        loss.backward()
        self._dql_net._optimizer.step()

    # TODO
    def Learn(self):
        pass

