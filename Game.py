from Minesweeper_Text_v0 import Minesweeper_Text_v0

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

_gamma = .997
_batch_size = 100
_lg_batch_size = 64
_memory_size = 10000
_learning_rate = 1e-4
_epsilon_start = 1.0
_epsilon_final = 0.01
_epsilon_decay = 0.99999

_solved_win_count = 25

_mode = 2
_difficulty = 1.0

# how many games to play before copying
_sync_target = 10000


# Perhpas the issues is the lack of symantic meaning to each of the numbers.
# therefore I should connect the cells, and their neighbours, and send in as a convolution

def main():
    if _mode == 0:
        # human
        from Minesweeper_v1 import Minesweeper_v1
        env = Minesweeper_v1(human=_human, difficulty=_difficulty)
        env.Play()
    elif _mode == 1:
        # q learning
        from BroomDQL import BroomDQL
        from BroomDQL import Memory
        from BroomDQL import Agent
        
        env = Minesweeper_Text_v0(_difficulty)
        device = torch.device("cuda")
        net = BroomDQL(
            env.observation_space.shape, env.action_space.n).to(device)
        target_net =BroomDQL(
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
            loss = CalculateQLoss(batch, net, target_net, device)
            loss.backward()
            optimizer.step()

            if consecutive_wins == _solved_win_count:
                print("solved!")
                torch.save(net.state_dict(), "minesweeper-best_%.0f.dat" % mean_reward)
                exit()
                
    elif _mode == 2:
        # binary classifier
        
        from BroomLG import BroomLG
        from BroomLG import Agent as LG_Agent
        from BroomLG import Memory as LG_Memory
        
        from BroomDQL import BroomDQL
        from BroomDQL import Memory as DQL_Memory
        from BroomDQL import AlternateAgent as DQL_Agent
        
        env = Minesweeper_Text_v0(_difficulty)
        device = torch.device('cuda')
        lg_net = BroomLG(9, 2).to(device)
        
        lg_memory = LG_Memory(_memory_size)
        lg_agent = LG_Agent(env, lg_memory)
        
        lg_optimizer = optim.Adam(lg_net.parameters(), lr=_learning_rate)
        lg_criterion = nn.CrossEntropyLoss()
        
        dql_net = BroomDQL(env.observation_space.shape, env.action_space.n).to(device)
        dql_target_net = BroomDQL(env.observation_space.shape, env.action_space.n).to(device)
        
        dql_memory = DQL_Memory(_memory_size)
        dql_agent = DQL_Agent(env, dql_memory)
        dql_optimizer = optim.Adam(dql_net.parameters(), lr=_learning_rate)
        
        
        epsilon = _epsilon_start
        total_rewards = []
        best_mean_reward = -9999
        steps = 0
        total_steps = 0
        games = 0
        solved_games = 0

        consecutive_wins = 0
        
        
        lg_map, lg_state = lg_agent.ReadBoard(lg_net, device)
        # I need to have the dql agent read the board from the lg agent
        # then I need to make a choice while keeping the lg_agents state as a factor
        # save those in a memory for the dql and save the selected spot and label in 
        # the lg_agent memory so that it can be trained on 
        # need to wait until the batch is large enough on both before training
        
        
        # 2020/06/23 I need to give two labels to the lg_net, 0 = no mine, 1 = mine
        # if no-mine > mine = 0, else = 1 for the board
        # Then they both need to be saved and placed into a memory
        # both the predictions and the labels need to be fed in
        # ex. pred = [0.11, 2.0]
        # ex. label = [1]
        while True:
            env_state, action, reward, done, done_reward, win = dql_agent.PlayStep(dql_net, lg_state, steps, epsilon, device)
            
            lg_map, lg_next_state, lg_predictions = lg_agent.ReadBoard(lg_net, device)
            
            # record experiences
            dql_agent.AddMemory(lg_state, action, reward, done, lg_next_state)
            lg_agent.AddMemory(lg_map, lg_state, lg_predictions, action, env_state)
            
            # update the curr state to the next state
            lg_state = lg_next_state
            
            if done_reward is not None:
                total_rewards.append(done_reward)
                # get mean of last 100 rewards
                mean_reward = np.mean(total_rewards[-100:])
                print("- games: %d, steps: %d, reward: %.3f, eps: %.2f, solved games: %d" %
                      (games, steps, mean_reward, epsilon, solved_games))

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(dql_net.state_dict(), "minesweeper-best_%.0f.dat" % mean_reward)
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
            if len(lg_memory) < _lg_batch_size or len(dql_memory) < _batch_size:
                continue
            
            if total_steps % _sync_target == 0:
                target_net.load_state_dict(net.state_dict())
            
            # optimize
            # lg first
            lg_optimizer.zero_grad()
            lg_batch = lg_memory.sample(_batch_size)
            
            loss = CalculateLGLoss(lg_batch, lg_criterion, lg_net, device)
            
            
            if done:
                lg_agent.Reset()
                dql_agent.Reset()
        
def CalculateLGLoss(batch, criterion, net, device='cpu'):
    inputs, outputs, labels = batch
    
        # TODO need to switch to the new full predicted list with a label to the 
        # correct answer
    inputs_t = torch.tensor(inputs).to(device)
    outputs_t = torch.tensor([outputs], dtype=torch.float32).to(device)
    labels_t = torch.tensor([labels], dtype=torch.int64).to(device)
    
    outs = net(inputs_t)
    
    # print(outs)
    print(outputs_t, len(outputs_t))
    print(labels_t, len(labels_t))
    
    return criterion(outputs_t, labels_t)

def CalculateQLoss(batch, net, target_net, device='cpu'):
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


def CalculateBinaryLoss():
    pass

if __name__ == "__main__":
    main()
