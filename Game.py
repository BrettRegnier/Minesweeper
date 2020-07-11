import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
from time import time
import math

_mode = 1
_difficulty = 1

# how many steps to play before copying



def main():
    if _mode == 0:
        # human
        from Minesweeper_v1 import Minesweeper_v1
        env = Minesweeper_v1(human=_human, difficulty=_difficulty)
        env.Play()
    elif _mode == 1:
        from Minesweeper_Text_v0 import Minesweeper_Text_v0
        from BroomDQL import Agent
        from Memory import Experience
        from Memory import Memory
        from Replay import Replay

        # hyperparameters
        memory_size = 1000
        learning_rate = 1e-4
        gamma = .99
        
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 0.999

        batch_size = 100
        sync_target = 150

        reset_threshold = 50 # maybe remove
        solved_win_count = 25 # maybe remove


        device = torch.device("cuda")
        env = Minesweeper_Text_v0(_difficulty)
        replay = Replay()
        memory = Memory(memory_size)

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

        while True:
            done = False
            win = False
            steps = 0
            accumulated_reward = 0

            state = env.reset(True)
            while not done:
                # take an action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = agent.Act(state, device)

                # perform the step
                next_state, reward, done, info = env.step(action)

                # record the memory if the agent didn't win/lose on first action
                # this allows it to converge slightly faster.
                exp = Experience(state, action, reward, done, next_state)
                if not (steps == 0 and done):
                    memory.append(exp)
                    
                state = next_state

                accumulated_reward += reward

                steps += 1
                total_steps += 1

                if len(memory) < batch_size:
                    continue

                batch = memory.sample(batch_size)
                agent.LearnBatch(batch, gamma, device)

            # after done
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
                print("{:<5}".format("win"), end="")
            else:
                print("{:<5}".format("lose"), end="")
            
            print(" - games: %d, steps: %d, reward: %.3f, eps: %.2f, wins: %d, loses: %d, solved games: %d" %
                      (games, steps, mean_reward, epsilon, wins, loses, solved_games), end=" ")
            if best_mean_reward < mean_reward:
                print("Best reward updated %.3f -> %.3f" %
                    (best_mean_reward, mean_reward), end="")
                best_mean_reward = mean_reward
            print("")

    if _mode == 3:
        from BroomA2C import BroomConvoA2C
        from BroomA2C import AgentA2C

        device = torch.device("cuda")
        env = Minesweeper_Text_v0(_difficulty)
        learning_rate = 0.0001
        actor_critic = BroomConvoA2C(env.observation_space.shape, env.action_space.n, lr=learning_rate)

        actor_critic.to(device)

        agent = AgentA2C(actor_critic)

        score_history = []
        n_epsiodes = 3500

        wins = 0
        loses = 0

        gamma = 0.99

        games = 0

        compress_v = np.vectorize(lambda a : (a+2)/11)

        while True:
            done = False
            win = False
            score = 0
            steps = 0
            state = env.reset(soft=True)
            SplitState(state)
            state_n = compress_v(state)
            state_t = torch.tensor([state_n], dtype=torch.float).to(device)
            while not done:
                steps += 1


                action = agent.Act(state_t)
                next_state, reward, done, misc = env.step(action)
                next_state_n = compress_v(next_state)
                next_state_t = torch.tensor([next_state_n], dtype=torch.float).to(device)
                
                reward_t = torch.tensor(np.array(reward), dtype=torch.float).to(device)
                # action_t = torch.tensor(np.array(action)).to(device)
                done_t = torch.tensor(done).to(device)


                win = misc['win']
                score += reward


                agent.Learn(state_t, reward_t, next_state_t, done_t, gamma)

                state_t = next_state_t

            games += 1


            if win:
                print("{:<7}".format("win - "), end="")
                wins += 1
            else:            
                print("{:<7}".format("lose - "), end="")
                loses += 1

            print("episode: %d, steps: %d, score: %.3f, wins: %d, loses: %d" % (games, steps, score, wins, loses))
            score_history.append(score)

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
