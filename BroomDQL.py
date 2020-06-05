
import random
import torch
import torch.nn as nn
import numpy as np

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
            nn.Linear(512, n_actions)
        )

        # neurons = input_shape[1] * input_shape[2]
        # neurons = input_shape[1] * 500
        # neurons = 400
        # self._fc = nn.Sequential(
        #     nn.Linear(input_shape[1], 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, n_actions)
        # )

    def forward(self, X):
        conv_out = self._conv(X).view(X.size()[0], -1)
        # print(conv_out)
        # exit()
        return self._fc(conv_out)
        #return self._fc(X)