
import random
import torch
import torch.nn as nn
import numpy as np
import collections

class Broom(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Broom, self).__init__()
        self._conv = nn.Sequential(
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

        out = self._conv(torch.zeros(1, *input_shape))
        conv_out_shape = int(np.prod(out.size()))
        print(conv_out_shape)
        # print(conv_out_shape); exit()
        self._fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.Linear(512, 512),
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

        #step
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