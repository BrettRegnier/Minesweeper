import torch
import numpy as np
from BroomSub import Agent
from Replay import UniformExperienceReplay
from Replay import Experience
from Minesweeper_Text_v0 import Minesweeper_Text_v0

def TransformState(state):
    state_np = np.array(state, copy=False)
    state_np = np.pad(state_np[0], 1)

    # create states
    # print(state_np)
    new_states = []
    for row in range(1, len(state_np) - 1):
        for column in range(1, len(state_np[row]) - 1):
            # sliding window
            tile_box = []
            for i in range(row-1, row+2):
                tile_row = []
                for j in range(column-1, column+2):
                    tile_row.append(state_np[i][j])
                tile_box.append(tile_row)

            # print(np.vstack(tile_box))
            new_states.append(tile_box)

    return new_states

lr = 1e-3
memory_size = 1000
batch_size = 64

env = Minesweeper_Text_v0(1)
device = torch.device('cuda')
agent = Agent(env.observation_space, env.action_space.n, lr, device)
memory = UniformExperienceReplay(memory_size)

while True:
    done = False
    steps = 0

    state = env.reset(soft=True)
    state = TransformState(state)
    while not done:
        action = agent.Act(state)

        next_state, reward, done, info = env.step(action)
        next_state = TransformState(next_state)

        exp = Experience([state[action]], action, reward, done, [next_state[action]])
        # if not (steps == 0 and done):
        memory.Append(exp)

        if len(memory) > batch_size:
            batch = memory.Sample(batch_size)
            agent.LearnBatch(batch, 0.99)
        
        # end 
        state = next_state

    agent.UpdateTarget()
    win = info['win']
    if win:
        print("{:<4}".format("win"), end="\n")
    else:            
        print("{:<4}".format("lose"), end="\n")


