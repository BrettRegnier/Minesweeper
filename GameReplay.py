import collections
import os


class GameReplay:
    def __init__(self):
        self._moves = collections.deque()

    def __len__(self):
        return len(self._moves)

    def append(self, move):
        self._move.append(move)

    def clear(self):
        self._moves.clear()

    def save(self, save_location, seed):
        save_string = str(seed) + "\n"

        for move in self._moves:
            save_string += str(move) + "\n"

        with open(save_location, 'w') as file:
            file.write(save_string)

    def load(self, save_location):
        pass
