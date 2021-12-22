import torchdrl.tools.Config as Config
from torchdrl.factories.AgentFactory import CreateQLearningAgent
from torchdrl.managers.RLManager import RLManager


from Minesweeper.env.Minesweeper_Text_v0 import Minesweeper_Text_v0
# from Minesweeper.env.Minesweeper_GUI_v1 import Minesweeper_GUI_v1
from Minesweeper.managers.BroomManager import BroomManager

from Minesweeper.wrappers.RandomFirstMove import RandomFirstMove

config = Config.Load("./config/dql_3_updated.json")

env = RandomFirstMove(Minesweeper_Text_v0(**config['env']['kwargs']))
agent = CreateQLearningAgent(config['agent'], env)
# env.reset()
# env.render(); exit()
# manager = BroomManager(env, agent)
# manager.TrainNoYield()

manager = RLManager(agent, **config['manager']['kwargs'])

manager.TrainNoYield();


