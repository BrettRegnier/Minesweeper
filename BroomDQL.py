
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
        # print(input_shape[0])
        self._conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 80, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))

        self._fc = nn.Sequential(
            nn.Linear(conv_out_shape, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_actions)
        )

        self._optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._fc(conv_out)

class Agent:
    def __init__(self, input_shape, n_actions, sync_target, lr, device):
        self._device = device
        self._dql_net = Network(input_shape, n_actions, lr).to(self._device)
        self._dql_target_net = Network(input_shape, n_actions, lr).to(self._device)
        self._loss_func = nn.MSELoss()
        self._sync_target = sync_target
        self._steps = 0


    @torch.no_grad()
    def PlayStep(self, net, steps, epsilon):
        done_reward = None
        win = False

        # choose random action for exploration
        if np.random.random() < epsilon:
            action = self._env.action_space.sample()
        else:
            current_state = np.array([self._state], copy=False)
            state_value = torch.FloatTensor(current_state).to(self._device)
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
    def Act(self, state):
        state_np = np.array([state], copy=False)
        state_t = torch.tensor(state_np).to(self._device)

        q_value = self._dql_net(state_t)
        value, action_value = torch.max(q_value, dim=1)
        action = int(action_value.item())    
        return action

    def LearnBatchPER(self, batch, is_weights, gamma):
        self._dql_net._optimizer.zero_grad()

        # calculate loss
        states, actions, next_states, rewards, dones = batch

        states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(self._device)
        next_states_t = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64).to(self._device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.bool).to(self._device)

        is_weights_t = torch.tensor(np.array(is_weights), dtype=torch.float32).to(self._device)


        state_action_values = self._dql_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self._dql_target_net(next_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_t


        loss = self._loss_func(state_action_values, expected_state_action_values)
        loss = (loss * is_weights_t).mean()
        loss.backward()
        self._dql_net._optimizer.step()

        errors = torch.abs(state_action_values - expected_state_action_values).cpu().detach().numpy()
        return errors

    @torch.no_grad()
    def GetError(self, state, action, next_state, reward, done, gamma):
        state_np = np.array([state], copy=False)
        state_t = torch.tensor(state_np).to(self._device)
        q_values = self._dql_net(state_t)
        action_value = q_values[0][action]
        
        next_state_np = np.array([next_state], copy=False)
        next_state_t = torch.tensor(next_state_np).to(self._device)
        next_q_values = self._dql_target_net(next_state_t)
        next_action_value = torch.max(next_q_values)

        expected_value = reward
        if not done:
            expected_value = next_action_value * gamma + reward

        error = abs(action_value - next_action_value).to("cpu")

        return error

    def Learn(self, state, action, reward, next_state, done):
        self._dql_net._optimizer.zero_grad()

        state_t = torch.tensor([state], dtype=torch.float32).detach().to(self._device).unsqueeze(0)
        next_state_t = torch.tensor([next_state], dtype=torch.float32).detach().to(self._device).unsqueeze(0)
        reward_t = torch.tensor(reward, dtype=torch.float32).detach().to(self._device)

        q_values = self._dql_net(state_t)
        next_q_values = self._dql_net(next_state_t)

        state_action_value = q_values[0][action]
        next_state_action_value = next_q_values[0].max()

        expected_value = reward_t + next_state_action_value * (1-done)

        loss = self._loss_func(state_action_value, next_state_action_value)

        loss.backward()

        self._dql_net._optimizer.step()

    def LearnBatch(self, batch, gamma):
        self._dql_net._optimizer.zero_grad()

        # calculate loss
        states, actions, rewards, dones, next_states = batch

        states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(self._device)
        next_states_t = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64).to(self._device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.bool).to(self._device)

        state_action_values = self._dql_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        sq = actions_t.unsqueeze(-1)
        test = self._dql_net(states_t).gather(1, sq)
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
