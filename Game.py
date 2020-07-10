from Minesweeper_Text_v0 import Minesweeper_Text_v0

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
from time import time

_gamma = .99
_batch_size = 16
_memory_size = 1000
_learning_rate = 1e-4
_epsilon_start = 1.0
_epsilon_final = 0.01
_epsilon_decay = 0.999

_reset_threshold = 50
_solved_win_count = 25

_mode = 3
_difficulty = 1.0

# how many steps to play before copying
_sync_target = 150



def main():
    if _mode == 0:
        # human
        from Minesweeper_v1 import Minesweeper_v1
        env = Minesweeper_v1(human=_human, difficulty=_difficulty)
        env.Play()
    elif _mode == 1:
        # q learning
        from Broom import Broom
        from Broom import Memory
        from Broom import Agent
        
        env = Minesweeper_Text_v0(_difficulty)
        device = torch.device("cuda")
        net = Broom(
            env.observation_space.shape, env.action_space.n).to(device)
        target_net = Broom(
            env.observation_space.shape, env.action_space.n).to(device)
        
        if os.path.isfile("./minesweeper-best.dat"):
            net.load_state_dict(torch.load("./minesweeper-best.dat"))
            target_net.load_state_dict(torch.load("./minesweeper-best.dat"))
            print("loaded")

        memory = Memory(_memory_size)
        agent = Agent(env, memory)
        epsilon = _epsilon_start

        optimizer = optim.Adam(net.parameters(), lr=_learning_rate)
        total_rewards = []
        best_mean_reward = -9999
        steps = 0
        total_steps = 0
        games = 0
        solved_games = 0
        
        wins = 0
        loses = 0

        consecutive_wins = 0

        # training loop
        while True:
            reward, misc = agent.PlayStep(net, steps, epsilon, device)
            win = misc['win']
            steps += 1
            total_steps += 1
            if reward is not None:
                if win:
                    consecutive_wins += 1
                    wins += 1
                    print('Win ', end="")
                else:
                    consecutive_wins = 0
                    loses += 1
                    print("Lose ", end="")
                games += 1
                
                # reset the game
                if games > _reset_threshold:
                    agent.Reset(soft=False)
                    
                total_rewards.append(reward)
                # get mean of last 100 rewards
                mean_reward = np.mean(total_rewards[-100:])
                print("- games: %d, steps: %d, reward: %.3f, eps: %.2f, wins: %d, loses: %d, solved games: %d" %
                      (games, steps, mean_reward, epsilon, wins, loses, solved_games))

                if best_mean_reward < mean_reward:
                    # torch.save(net.state_dict(), "minesweeper-best_%.0f.dat" % mean_reward)
                    print("Best reward updated %.3f -> %.3f" %
                          (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward


                steps = 0
                epsilon = max(_epsilon_final, epsilon * _epsilon_decay)
            
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
    
    elif _mode == 2:
        from Broom import RandomTuner
        
        env = Minesweeper_Text_v0(_difficulty)
        device = torch.device("cuda")
    
        tuner = RandomTuner(env)
        
        tuner.RunTuning(device)

    elif _mode == 3:
        from BroomA2C import BroomA2C
        from BroomA2C import AgentA2C

        device = torch.device("cuda")
        env = Minesweeper_Text_v0(_difficulty)

        actor = BroomA2C(env.observation_space.shape, env.action_space.n, lr=0.00001)
        critic = BroomA2C(env.observation_space.shape, 1, lr=0.0005)

        actor.to(device)
        critic.to(device)

        agent = AgentA2C(actor, critic)

        score_history = []
        n_epsiodes = 3500

        wins = 0
        loses = 0

        gamma = 0.99

        games = 0
        while True:
            done = False
            win = False
            score = 0
            steps = 0
            state = env.reset()
            while not done:
                steps += 1

                state_np = np.reshape(state, (-1))
                state_t = torch.FloatTensor(state_np).to(device)

                action = agent.Act(state_t)
                next_state, reward, done, misc = env.step(action)

                next_state_np = np.reshape(next_state, (-1))
                next_state_t = torch.FloatTensor(next_state_np).to(device)
                
                reward_t = torch.tensor(np.array(reward)).to(device)
                # action_t = torch.tensor(np.array(action)).to(device)
                done_t = torch.tensor(done).to(device)


                win = misc['win']
                score += reward


                agent.Learn(state_t, reward_t, next_state_t, done_t, gamma)

                state = next_state

            games += 1


            if win:
                print("{:<7}".format("win - "), end="")
                wins += 1
            else:            
                print("{:<7}".format("lose - "), end="")
                loses += 1

            print("episode: %d, steps: %d, score: %.3f, wins: %d, loses: %d" % (games, steps, score, wins, loses))
            score_history.append(score)

    
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

def PrintStatus():
    pass

if __name__ == "__main__":
    main()
