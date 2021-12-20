import torch
import random
import numpy as np

from torchdrl.factories.AgentFactory import AgentFactory
from torchdrl.managers.RLManager import RLManager
import torchdrl.tools.Config as Config

from minesweeper.cli.Minesweeper_Text_v0 import Minesweeper_Text_v0
from minesweeper.gui.Minesweeper_v1 import Minesweeper_v1

config = Config.Load("./config/minesweeper_dql.json")

num_envs = 1
difficulty = 1

envs = []
for i in range(num_envs):
    envs.append(Minesweeper_Text_v0(difficulty))

# set the seed
seed = 0
if seed >= 0:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Seed envs
    for i, env in enumerate(envs):
        env.seed(seed+i)

q_learning_agent = AgentFactory.CreateQLearningAgent(config['q_learning_agent'], envs)

manager = RLManager(q_learning_agent, **config['manager']['kwargs'])

manager.TrainNoYield()