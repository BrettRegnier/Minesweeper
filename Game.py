from Minesweeper_v1 import Minesweeper_v1
from Minesweeper_Text_v0 import Minesweeper_Text_v0
import BroomDQL
import Memory
import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

_gamma = .997
_batch_size = 1000
_memory_size = 10000
_learning_rate = 1e-4
_epsilon_start = 1.0
_epsilon_final = 0.01
_epsilon_decay = 0.99999

_solved_win_count = 25

_human = False
_difficulty = 1.0

# how many games to play before copying
_sync_target = 10000


# Perhpas the issues is the lack of symantic meaning to each of the numbers.
# therefore I should connect the cells, and their neighbours, and send in as a convolution

def main():
    if _human:
        env = Minesweeper_v1(human=_human, difficulty=_difficulty)
        env.Play()
    else:
        env = Minesweeper_Text_v0(_difficulty)
        device = torch.device("cuda")
        net = BroomDQL.BroomDQL(
            env.observation_space.shape, env.action_space.n).to(device)
        target_net = BroomDQL.BroomDQL(
            env.observation_space.shape, env.action_space.n).to(device)

        if os.path.isfile("./minesweeper-best.dat"):
            net.load_state_dict(torch.load("./minesweeper-best.dat"))
            target_net.load_state_dict(torch.load("./minesweeper-best.dat"))
            print("loaded")

        memory = Memory.Memory(_memory_size)
        agent = Agent.Agent(env, memory)
        epsilon = _epsilon_start

        optimizer = optim.Adam(net.parameters(), lr=_learning_rate)
        total_rewards = []
        best_mean_reward = -9999
        steps = 0
        total_steps = 0
        games = 0
        solved_games = 0

        consecutive_wins = 0

        # training loop
        while True:
            reward, win = agent.PlayStep(net, steps, epsilon, device)
            if reward is not None:
                total_rewards.append(reward)
                # get mean of last 100 rewards
                mean_reward = np.mean(total_rewards[-100:])
                print("- games: %d, steps: %d, reward: %.3f, eps: %.2f, solved games: %d" %
                      (games, steps, mean_reward, epsilon, solved_games))

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), "minesweeper-best_%.0f.dat" % mean_reward)
                    print("Best reward updated %.3f -> %.3f" %
                          (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward

                if win:
                    consecutive_wins += 1
                else:
                    consecutive_wins = 0

                steps = 0
                games += 1
                epsilon = max(_epsilon_final, epsilon * _epsilon_decay)
            
            steps += 1
            total_steps += 1
            if len(memory) < _batch_size:
                continue

            if total_steps % _sync_target == 0:
                target_net.load_state_dict(net.state_dict())
                # print("load target")

            optimizer.zero_grad()
            batch = memory.sample(_batch_size)
            loss = CalculateLoss(batch, net, target_net, device)
            loss.backward()
            optimizer.step()

            if consecutive_wins == _solved_win_count:
                print("solved!")
                torch.save(net.state_dict(), "minesweeper-best_%.0f.dat" % mean_reward)
                exit()


def CalculateLoss(batch, net, target_net, device='cpu'):
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

    expected_state_action_values = next_state_values * _gamma + rewards_d

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    main()
