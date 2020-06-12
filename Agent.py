import collections
import torch
import numpy as np


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'next_state'])

# move into DQN??

class Agent:
    def __init__(self, env, memory):
        self._env = env
        self._memory = memory
        self.Reset()

    def Reset(self, soft=False):
        self._state = self._env.reset(soft=soft)
        self._total_reward = 0

# add a tracking to what match this memory was
# extend the memory tuple object
    @torch.no_grad()
    def PlayStep(self, net, steps, epsilon=0.0, device="cpu"):
        done_reward = None
        win = False

        # choose random action for exploration
        if np.random.random() < epsilon:
            action = self._env.action_space.sample()
        else:
            current_state = np.array([self._state], copy=False)
            state_value = torch.FloatTensor(current_state).to(device)
            q_value = net(state_value)
            value, action_value = torch.max(q_value, dim=1)
            action = int(action_value.item())

        #step
        next_state, reward, done, win = self._env.step(action)
            

        # self._env.render()
        self._total_reward += reward

        experience = Experience(self._state, action, reward, done, next_state)
        if not (steps == 0 and done):
            self._memory.append(experience)
        self._state = next_state

        if done:
            done_reward = self._total_reward
            self.Reset()
        return done_reward, win
