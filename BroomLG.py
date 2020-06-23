import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import collections

class BroomLG(nn.Module):
    def __init__(self, input_shape, action_space):
        super(BroomLG, self).__init__()
        
        # self._conv = nn.Sequential(
        #     nn.Conv2d(input_shape, 12, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        
        # out = self._conv(torch.zeros(1, *input_shape))
        # conv_out_shape = int(np.prod(out.size()))
        
        self._fc = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.Tanh(),
            nn.Linear(32, action_space),
            nn.ReLU() 
        )
        
    def forward(self, X):
        # conv_out = self._conv(X).view(X.size()[0], -1)
        # return self._fc(conv_out)
        return self._fc(X)
      
Experience = collections.namedtuple('experience', field_names=['input', 'output', 'label'])  
class Agent:
    def __init__(self, env, memory):
        self._env = env
        self._memory = memory
        self.Reset()
    
    def Reset(self, soft=False):
        done = True
        while done:
            self._env.reset(soft=soft)
            action = self._env.action_space.sample()
            self._state, _, done, _ = self._env.step(action)
            
        self._total_reward = 0  
    
    @torch.no_grad()
    def ReadBoard(self, net, device='cpu'):
        done_reward = None
        win = False
        
        # build convo map input
        lg_map = []
        # print(len(self._state[0]))
        rows = len(self._state[0])
        columns = len(self._state[0][0])
        for i, row in enumerate(self._state[0]):
            lg_map.append([])
            for j, tile in enumerate(row):
                lg_map[i].append([])
                for k in range(i-1, i+2):
                    if k < 0 or k >= rows:
                        lg_map[i][j].append(-1)
                        lg_map[i][j].append(-1)
                        lg_map[i][j].append(-1)
                        continue
                    for n in range(j-1, j+2):
                        if k == i and n == j:
                            lg_map[i][j].append(tile)
                            continue
                        if n < 0 or n >= columns:
                            lg_map[i][j].append(-1)
                            continue
                        
                        lg_map[i][j].append(self._state[0][k][n])      
        
        # feed convo map into the nn with (no grad so that it only gets updated after an action?)
        # figure out how to update this nn based on a single answer.
        
        # feed into logistic regression
        c = 0
        
        # 0 means no mine - 1 means definitely mine
        lg_board = []
        lg_state = []
        for i, row in enumerate(lg_map):
            lg_board.append([])
            for j, group in enumerate(row):
                c += 1
                tile_v = net(torch.tensor(np.array(group, dtype=np.float32)).to(device))
                
                if torch.argmax(tile_v) == 0:
                    # not a mine
                    lg_board[i].append(0)
                else:
                    # is mine, place probability
                    lg_board[i].append(tile_v[1])
                lg_state[i].append(tile_v)
              
        lg_board = np.array(lg_board, dtype=np.float32, copy=False)
        lg_board = np.expand_dims(lg_board, axis=0)
        
        # print(lg_board)
        # print(c)
        return lg_map, lg_board, lg_predictions
        
    # lg map is also just the current env state
    def AddMemory(self, lg_map, lg_state, action, env_next_state):
        rows = len(lg_map)
        columns = len(lg_map[0])
        
        # print(lg_map) 
        # print(rows)
        # print(columns)
        
        row = action // columns    
        column = action % columns
        
        ans = env_next_state[0][row][column]
        # print(env_next_state)
        # print("action", action, "row", row, "column", column)
        # print("value of tile", ans)
        
        y_label = 0
        # mine tile == 10
        if ans == 10:
            y_label = 1
        
        x_input = np.array(lg_map[row][column], dtype=np.float32)
        
        # #################################
        # TODO add predictions into the experience inplace of the y_out
        #-------------------------------------#
        y_out = lg_state[0][row][column]
        
        # print("input", x_input)  
        # print("prediction", y_out, "label", y_label)
        
        self._memory.append(Experience(x_input, y_out, y_label))
            
class Memory:
    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self._buffer)
        
    def append(self, experience):
        self._buffer.append(experience)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        
        inputs = []
        outputs = []
        labels = []
        for i in indices:
            inputs.append(self._buffer[i].input)
            outputs.append(self._buffer[i].output)
            labels.append(self._buffer[i].label)
                
        # print(inputs)
        # print(labels)
        # print(outputs)
        return np.array(inputs), np.array(outputs), np.array(labels)
    