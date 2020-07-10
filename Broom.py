
import random
import torch
import torch.nn as nn
import numpy as np
import collections


class Broom(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Broom, self).__init__()
        conv = nn.Sequential(
            # nn.Conv2d(in_channels=input_shape[0], out_channels=240, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=240, out_channels=160, kernel_size=3, stride=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3),

            # nn.Conv2d(in_channels=160, out_channels=80, kernel_size=2),
            # nn.ReLU()

            nn.Conv2d(input_shape[0], 80, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, stride=1),
            nn.ReLU()
        )

        out = conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))
        # print(conv_out_shape)
        # print(conv_out_shape); exit()
        fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.Linear(512, 512),
            nn.Linear(512, n_actions)
        )
        
        self.SetLayers(conv, fc)
        
    def SetLayers(self, conv, fc):
        self._conv = conv
        self._fc = fc

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        return self._fc(conv_out)


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'next_state'])


class Agent:
    def __init__(self, env, memory):
        self._env = env
        self._memory = memory
        self._replay = Replay()
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

        # step
        next_state, reward, done, win = self._env.step(action)

        experience = Experience(self._state, action, reward, done, next_state)
        if not (steps == 0 and done):
            self._memory.append(experience)

        self._total_reward += reward
        self._state = next_state

        if done:
            done_reward = self._total_reward
            self.Reset(soft=True)
            if win:
                self._replay.save("./replays", self._env._seed)
                self._replay.clear()

        return done_reward, win


class Memory:
    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def append(self, experience):
        self._buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(
            len(self._buffer), batch_size, replace=False)

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


class Replay:
    def __init__(self):
        self._moves = collections.deque()

    def __len__(self):
        return len(self._moves)

    def append(self, move):
        self._move.append(move)

    def clear(self):
        self._moves.clear()

    def save(self, save_location, seed):
        save_string = str(seed) + "\n"

        for move in self._moves:
            save_string += str(move) + "\n"

        with open(save_location, 'w') as file:
            file.write(save_string)

    def load(self, save_location):
        pass

from time import time
import torch.optim as optim
import os
class RandomTuner:
    def __init__(self, env):
        self._storage = {}
        
        self._hyper = {}
        self._hyper["wins"] = 0
        self._hyper["loses"] = 0
        self._hyper["win-loss_percent"] = 0
        self._hyper["input_shape"] = env.observation_space.shape
        self._hyper["n_actions"] = env.action_space.n
        self._hyper["gamma"] = .99
        self._hyper["batch_size"] = 16
        self._hyper["memory_size"] = 1000
        self._hyper["learning_rate"] = 1e-4
        self._hyper["epsilon_start"] = 1.0
        self._hyper["epsilon_final"] = 0.01
        self._hyper["epsilon_decay"] = 0.999
        self._hyper["reset_threshold"] = 5
        self._hyper["solved_win_count"] = 25
        self._hyper["sync_target"] = 100
        
        self._env = env
        
        self._num_tuning = 10
        
    def RandomTune(self):
        # logic on changing the parameters
        # start with Neurons
        filters = random.randint(70, 160)
        kernel_size = random.randint(3, 4)
        stride = random.randint(1, 2)
        padding = random.randint(1, 2)

        conv = nn.Sequential(
            nn.Conv2d(in_channels=self._hyper['input_shape'][0], out_channels=filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size)
            )
            
        out = conv(torch.zeros(1, *self._hyper['input_shape']))
        conv_out_shape = int(np.prod(out.size()))
        
        modules = []
        fc_layers = random.randint(2, 6)
        prev_neurons = conv_out_shape
        for layer in range(fc_layers):
            neurons = random.randint(200, 1024)
            modules.append(nn.Linear(in_features=prev_neurons, out_features=neurons))
            prev_neurons = neurons
        
        modules.append(nn.Linear(in_features=prev_neurons, out_features=self._hyper["n_actions"]))
        
        fc = nn.Sequential(*modules)
        
        net = Broom(self._hyper['input_shape'], self._hyper['n_actions'])
        target_net = Broom(self._hyper['input_shape'], self._hyper['n_actions'])
        
        net.SetLayers(conv, fc)
        target_net.SetLayers(conv, fc)
        
        return net, target_net
        
    def RunTuning(self, device):
        for _ in range(self._num_tuning):
            self.Reset()
            
            net, target_net = self.RandomTune()
            net.to(device)
            target_net.to(device)
        
            optimizer = optim.Adam(net.parameters(), lr=self._hyper['learning_rate'])
            memory = Memory(self._hyper["memory_size"])
            agent = Agent(self._env, memory)
            
            epsilon = self._hyper['epsilon_start']
            
            total_rewards = collections.deque(maxlen=1000)
            best_mean_reward = -9999
        
            steps = 0
            total_steps = 0
            games = 0
            solved_games = 0
            
            wins = 0
            loses = 0

            consecutive_wins = 0
            
            # training loop
            while games < 3500:
                reward, misc = agent.PlayStep(net, steps, epsilon, device)
                win = misc['win']
                steps += 1
                total_steps += 1
                
                if reward is not None:
                    if win:
                        consecutive_wins += 1
                        wins += 1
                        print("Win ", end="")
                    else:
                        consecutive_wins = 0
                        loses += 1
                        print("Lose ", end="")
                    games += 1
                    
                    # shuffle the board
                    if games > self._hyper['reset_threshold']:
                        agent.Reset(soft=False)
                
                    total_rewards.append(reward)
                    
                    
                    # get mean of last 100 rewards
                    mean_reward = np.mean(list(total_rewards)[-100:])
                    print("- games: %d, steps: %d, reward: %.3f, eps: %.2f, wins: %d, loses: %d, solved games: %d" %
                        (games, steps, mean_reward, epsilon, wins, loses, solved_games))
                        
                    if best_mean_reward < mean_reward:
                        print("Best reward updated %.3f -> %.3f" %
                          (best_mean_reward, mean_reward))
                        best_mean_reward = mean_reward
                
                    steps = 0
                    epsilon = max(self._hyper['epsilon_final'], epsilon * self._hyper['epsilon_decay'])
                    
                    
                if len(memory) >= self._hyper['batch_size']:
                    if total_steps % self._hyper['sync_target'] == 0:
                        target_net.load_state_dict(net.state_dict())
                    
                    # optimize    
                    optimizer.zero_grad()
                    batch = memory.sample(self._hyper['batch_size'])
                    loss = self.CalculateLoss(batch, net, target_net, device)
                    loss.backward()
                    optimizer.step()
            
            self._hyper['wins'] = wins
            self._hyper['loses'] = loses
            self._hyper['win-loss_percent'] = round(wins / (wins+loses), 2)
            self.Store(net, mean_reward)
            self.Save(net, mean_reward)
    
        
    def CalculateLoss(self, batch, net, target_net, device='cpu'):
        states, actions, rewards, dones, next_states = batch

        states_d = torch.tensor(np.array(states, copy=False)).to(device)
        next_states_d = torch.tensor(
            np.array(next_states, copy=False)).to(device)
        actions_d = torch.LongTensor(np.array(actions)).to(device)
        rewards_d = torch.FloatTensor(np.array(rewards)).to(device)
        dones_d = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_d).gather(
            1, actions_d.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = target_net(next_states_d).max(1)[0]
            next_state_values[dones_d] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self._hyper['gamma'] + rewards_d

        return nn.MSELoss()(state_action_values, expected_state_action_values)
    
    def Reset(self):
        self._env.reset(soft=False)
    
    def Train(self, device):
        pass
    
    def Test(self, device):
        pass   
    
    def Store(self, net, reward):
        self._storage["reward"] = reward
        self._storage["net"] = net
        self._storage["parameters"] = self._hyper
    
    def Save(self, net, reward):
        if self._storage == 0:
            self.Store(net, reward)
            
        save_dir = "./models/broom/%.3f/" % reward
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(save_dir + "parameters.txt", "w") as file:
            for key in self._storage:
                file.write(str(key) + ":" + str(self._storage[key]) + "\n")
            
        
        
