import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
from time import time
import math

_mode = 1
_difficulty = 1

def main():
    if _mode == 0:
        # human
        from minesweeper.gui.Minesweeper_v1 import Minesweeper_v1
        env = Minesweeper_v1(human=True, difficulty=_difficulty)
        env.Play()
    elif _mode == 1:
        from minesweeper.cli.Minesweeper_Text_v0 import Minesweeper_Text_v0
        from BroomDQL import Agent
        from Replay import Experience
        from Replay import UniformExperienceReplay
        from Replay import PrioritizedExperienceReplay
        from GameReplay import GameReplay

        # hyperparameters
        memory_size = 2000
        learning_rate = 1e-4
        gamma = .99
        
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 0.99

        batch_size = 100
        sync_target = 150

        reset_threshold = 50 # maybe remove
        solved_win_count = 25 # maybe remove


        device = torch.device("cuda")
        env = Minesweeper_Text_v0(_difficulty)
        replay = GameReplay()
        memory = UniformExperienceReplay(memory_size)
        # memory = PrioritizedExperienceReplay(memory_size, 0.6)

        # declare agent
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
        agent = Agent(input_shape, n_actions, sync_target, learning_rate, device)
        epsilon = epsilon_start

        total_rewards = []
        best_mean_reward = -math.inf

        total_steps = 0
        
        games = 0
        wins = 0
        loses = 0
        consecutive_wins = 0
        solved_games = 0

        soft_reset_game = False

        while True:
            done = False
            win = False
            steps = 0
            accumulated_reward = 0

            state = env.reset(soft_reset_game)
            while not done:
                # take an action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = agent.Act(state)

                # perform the step
                next_state, reward, done, info = env.step(action)

                # record the memory if the agent didn't win/lose on first action
                # this allows it to converge slightly faster.
                exp = Experience(state, action, reward, done, next_state)

                # error = agent.GetError(state, action, next_state, reward, done, gamma)
                # exp = (error, (state, action, next_state, reward, done))

                if not (steps == 0 and done):
                    memory.Append(exp)
                    
                state = next_state

                accumulated_reward += reward

                steps += 1
                total_steps += 1

                if len(memory) < batch_size:
                    continue

                batch = memory.Sample(batch_size)
                agent.LearnBatch(batch, gamma)

                # batch, indices, is_weights = memory.Sample(batch_size)
                # errors = agent.LearnBatchPER(batch, is_weights, gamma)

                # for i in range(batch_size):
                #     memory.Update(indices[i], errors[i])

            # after done
            agent.UpdateTarget()
            games += 1
            win = info['win']
            if win:
                consecutive_wins += 1
                wins += 1
            else:
                consecutive_wins = 0
                loses += 1

            total_rewards.append(accumulated_reward)
            total_rewards = total_rewards[-100:]
            mean_reward = np.mean(total_rewards)

            epsilon = max(epsilon_final, epsilon * epsilon_decay)

            if win:
                print("{:<4}".format("win"), end="")
            else:
                print("{:<4}".format("lose"), end="")
            
            print(" - games: %d, steps: %d, reward: %.3f, eps: %.2f, wins: %d, loses: %d, solved games: %d" %
                      (games, steps, mean_reward, epsilon, wins, loses, solved_games), end=" ")
            if best_mean_reward < mean_reward:
                print("Best reward updated %.3f -> %.3f" %
                    (best_mean_reward, mean_reward), end="")
                best_mean_reward = mean_reward
            print("")

            if consecutive_wins == solved_win_count:
                solved_games += 1
                print("solved"); exit()
    if _mode == 2:
        from BroomA2C import BroomConvoA2C
        from BroomA2C import AgentA2C
        from Replay import UniformExperienceReplay
        from Replay import Experience
        from minesweeper.cli.Minesweeper_Text_v0 import Minesweeper_Text_v0

        learning_rate = 0.0001
        gamma = 0.99

        device = torch.device("cuda")
        env = Minesweeper_Text_v0(_difficulty)
        agent = AgentA2C(env.observation_space.shape, env.action_space.n, learning_rate, gamma, device)
        memory = UniformExperienceReplay(1000)

        total_rewards = []

        n_epsiodes = 3500
        solved_win_count = 25

        batch_size = 64

        total_steps = 0

        games = 0
        wins = 0
        loses = 0
        consecutive_wins = 0
        solved_games = 0

        compress_v = np.vectorize(lambda a : (a+2)/11)

        while True:
            done = False
            win = False
            steps = 0
            accumulated_reward = 0

            state = env.reset(soft=True)
            # SplitState(state)
            # state_n = compress_v(state)
            while not done:
                action, action_probs = agent.Act(state)
                next_state, reward, done, info = env.step(action)

                accumulated_reward += reward

                exp = Experience(state, action, reward,done, next_state)

                if not (steps == 0 and done):
                    memory.Append(exp)

                steps += 1
                total_steps += 1

                if len(memory) > batch_size:
                    batch = memory.Sample(batch_size)
                    agent.LearnBatch(batch)
                state = next_state

            # after done
            games += 1
            win = info['win']

            total_rewards.append(accumulated_reward)
            total_rewards = total_rewards[-100:]
            mean_reward = np.mean(total_rewards)

            if win:
                print("{:<4}".format("win"), end="")
                consecutive_wins += 1
                wins += 1
            else:            
                print("{:<4}".format("lose"), end="")
                consecutive_wins = 0
                loses += 1

            print(" - games: %d, steps: %d, reward: %.3f, wins: %d, loses: %d, solved games: %d" %
                      (games, steps, mean_reward, wins, loses, solved_games))

            if consecutive_wins == solved_win_count:
                solved_games += 1
                print("solved"); exit()

# TODO deprecate?
def SplitState(state):
    state = np.pad(state[0], 1)
    new_state = []

    for i in range(1, len(state) - 1):
        for j in range(1, len(state[i]) - 1):
            tiles = []
            for k in range(i-1, i + 2):
                row = []
                for l in range(j-1, j + 2):
                    tile = state[k][l]
                    row.append(tile)
                tiles.append(row)

            new_state.append(tiles)

    new_state = np.array(new_state)

    return new_state

def PrintStatus():
    pass

if __name__ == "__main__":
    main()
