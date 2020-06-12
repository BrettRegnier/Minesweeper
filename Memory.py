import collections
import numpy as np


class Memory:
    def __init__(self, capacity):
        self._buffer = collections.deque()
        self._game_buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self._game_buffer)

    def append(self, experience):
        self._buffer.append(experience)
        if experience.done:
            self._game_buffer.append(self._buffer.copy())
            self._buffer.clear()

    # i might need to change this so each match is instead a full match instead
    def sample(self, batch_size):
        indices = np.random.choice(
            len(self._game_buffer), batch_size, replace=False)

        
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for idx in indices:
            game = self._game_buffer[idx]
            for exp in game:
                states.append(exp.state)
                actions.append(exp.action)
                rewards.append(exp.reward)
                dones.append(exp.done)
                next_states.append(exp.next_state)

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.int8), np.array(next_states)
