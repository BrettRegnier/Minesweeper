import torch
import random
import numpy as np

from torchdrl.factories.AgentFactory import AgentFactory
from torchdrl.managers.RLManager import RLManager
import torchdrl.tools.Config as Config

from Game.CLI.Minesweeper_Text_v0 import Minesweeper_Text_v0
from Game.GUI.Minesweeper_v1 import Minesweeper_v1

config = Config.Load("./configurations/minesweeper_dql.json")

envs = []
for i in range(1):
    envs.append(gym.make("CartPole-v0"))

    

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
class Broom: 
    def __init__(self):
        pass

    def Train():
        pass

    def Test():
        pass