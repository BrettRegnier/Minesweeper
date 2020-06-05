import collections
import numpy as np


class Memory:
    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self._buffer)

    def append(self, experience):
        self._buffer.append(experience)

    # i might need to change this so each match is instead a full match instead
    def sample(self, batch_size):
        indices = np.random.choice(
            len(self._buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for idx in indices:
            x = self._buffer[idx]
            states.append(self._buffer[idx].state)
            actions.append(self._buffer[idx].action)
            rewards.append(self._buffer[idx].reward)
            dones.append(self._buffer[idx].done)
            next_states.append(self._buffer[idx].next_state)

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.int8), np.array(next_states)
