
import random
import torch
import torch.nn as nn

class BroomDQL(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BroomDQL, self).__init__()

        self.fc = nn.Sequential(
            # nn.Sigmoid(),
            nn.Linear(input_shape[1], input_shape[1]*8),
            nn.ReLU(),
            nn.Linear(input_shape[1]*8, n_actions)
        )

    def forward(self, X):
        return self.fc(X)