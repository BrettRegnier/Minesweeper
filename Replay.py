import collections
import numpy as np
import numpy
import random

class SumTree:
    def __init__(self, capacity):
        self._capacity = capacity

        # total nodes
        self._tree = np.zeros(2 * capacity - 1)

        # data is stored on the leaf nodes
        self._data = np.zeros(capacity, dtype=object)
        
        self._next_idx = 0
        self._n_entries = 0
    
    def _Retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if s <= self._tree[left]:
            return self._Retrieve(left, s)
        else:
            return self._Retrieve(right, s - self._tree[left])


    def Retrieve(self, idx, sample):
        left_idx = idx * 2 + 1
        right_idx = left_idx + 1

        while not left_idx >= len(self._tree):
            if sample <= self._tree[left_idx]:
                idx = left_idx
            else:
                sample -= self._tree[left_idx]
                idx = right_idx

            left_idx = idx * 2 + 1
            right_idx = left_idx + 1

        return idx

    def Total(self):
        return self._tree[0]

    def Add(self, priority, data):
        idx = self._next_idx + self._capacity - 1

        self._data[self._next_idx] = data
        self.Update(idx, priority)

        self._next_idx += 1

        # if we are past the capacity begin overwriting
        if self._next_idx >= self._capacity:
            self._next_idx = 0

        # increment the num of entries so far.
        if self._n_entries < self._capacity:
            self._n_entries += 1

    def Update(self, idx, priority):
        change = priority - self._tree[idx]
        self._tree[idx] = priority

        # propogate
        while idx != 0:
            idx = (idx - 1) // 2
            self._tree[idx] += change

    def Get(self, sample):
        idx = self.Retrieve(0, sample)
        # idx_2 = self.Retrieve(0, sample)

        # print(idx, idx_2)
        data_idx = idx - self._capacity + 1

        return (idx, self._tree[idx], self._data[data_idx])

# experience is of form (state, action, reward, next_state)
class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha):
        self._tree = SumTree(capacity)
        self._capacity = capacity

        # alpha
        self._alpha = alpha

        # small value to prevent edge cases
        self._epsilon = 0.01

        self._beta = 0.4
        self._beta_increment = 0.001

    def __len__(self):
        return self._tree._n_entries

    def GetPriority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    def Append(self, experience):
        error = experience[0]
        exp = experience[1]
        priority = self.GetPriority(error)
        self._tree.Add(priority, exp)

    def Sample(self, batch_size):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        
        batch = None

        indices = []
        segment = self._tree.Total() / batch_size
        priorities = []

        # TODO see how this works...
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            sample = np.random.uniform(a, b)
            idx, priority, data = self._tree.Get(sample)
            priorities.append(priority)

            states.append(data[0])
            actions.append(data[1])
            next_states.append(data[2])
            rewards.append(data[3])
            dones.append(data[4])

            indices.append(idx)

        states_np = np.array(states)
        actions_np = np.array(actions)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.int8)
        batch = (states_np, actions_np, next_states_np, rewards_np, dones_np)
        
        #this is the probability of sampling transition in the paper
        # dividing all of the priorities uniformly
        sampling_probs = priorities / self._tree.Total()

        #importance sampling = is
        is_weight = (1/self._tree._n_entries * sampling_probs) ** -self._beta
        is_weight = 1/is_weight

        # update beta
        self._beta = np.min([1., self._beta + self._beta_increment])

        return batch, indices, is_weight

    def Update(self, idx, error):
        priority = self.GetPriority(error)
        self._tree.Update(idx, priority)


Experience = collections.namedtuple('experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'next_state'])


class UniformExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def Append(self, experience):
        self.buffer.append(experience)

    def Sample(self, batch_size):
        indices = np.random.choice(
            len(self.buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for idx in indices:
            x = self.buffer[idx]
            states.append(self.buffer[idx].state)
            actions.append(self.buffer[idx].action)
            rewards.append(self.buffer[idx].reward)
            dones.append(self.buffer[idx].done)
            next_states.append(self.buffer[idx].next_state)

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.int8), np.array(next_states)
