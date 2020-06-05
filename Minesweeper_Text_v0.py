import gym
from gym import spaces
import numpy as np
import random

import collections

UNREVEALED_TILE = 9
MINE_TILE = 10

STATE = 0
NEIGHBOURS = 1
NUM_MINES = 2
IS_MINE = 3


class Minesweeper_Text_v0(gym.Env):
    def __init__(self, difficulty):
        super(Minesweeper_Text_v0, self).__init__()
        self._rows = int(6 * difficulty + 2)
        self._columns = int(11 * difficulty - difficulty**2)
        self._mines = int(((0.006895 * difficulty**2) + 0.013045 *
                           difficulty + 0.10506) * (self._rows * self._columns))

        self._board = None
        self._unrevealed_remaining = UNREVEALED_TILE

        self._first_action = True

        self.action_space = spaces.Discrete(self._rows * self._columns)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(1, self._rows, self._columns), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=10, shape=(
        #     1, self._rows * self._columns), dtype=np.int32)

    def step(self, action):
        tile = self._board[action]
        done = False
        state = None
        win = False
        reward = -0.3

        # check if mine
        if tile[IS_MINE] == True:
            done = True
            reward = -1
            tile[STATE] = MINE_TILE
            print("lose", end=" ")

        # check if unrevealed
        if tile[STATE] == UNREVEALED_TILE and not done:
            reward = 0.7

            # reveal all neighbouring tiles that can be
            queue = [tile]
            while len(queue) > 0:
                queue_tile = queue.pop(0)
                num_mines = queue_tile[NUM_MINES]
                if queue_tile[STATE] == UNREVEALED_TILE:
                    queue_tile[STATE] = num_mines
                    self._unrevealed_remaining -= 1
                if queue_tile[NUM_MINES] == 0 and not queue_tile[IS_MINE]:
                    for n_idx in queue_tile[NEIGHBOURS]:
                        if n_idx != -1:
                            adj = self._board[n_idx]
                            if adj[STATE] == UNREVEALED_TILE:
                                queue.append(adj)

        # check win condition
        if self._unrevealed_remaining == 0:
            done = True
            win = True
            reward = 1
            print("--win--", end=" ")

        if self._first_action and done:
            reward = 0
        self._first_action = False

        state = self.State()

        # print(state, reward, done, win)
        return state, reward, done, win

    def reset(self, soft=True):
        self._unrevealed_remaining = (self._rows * self._columns) - self._mines
        self._first_action = True
        if soft and self._board is not None:
            for tile in self._board:
                tile[STATE] = UNREVEALED_TILE
        else:
            self._board = []
            
            mine_indices = []
            to_make_mine = self._mines
            choices = [c for c in range(self._rows * self._columns)]
            while to_make_mine > 0:
                idx = random.choice(choices)
                mine_indices.append(idx)
                choices.remove(idx)
                to_make_mine -= 1

            for row in range(self._rows):
                for column in range(self._columns):
                    neighbours = []
                    neighbouring_mines = 0
                    for i in range(row-1, row+2):
                        if i < 0 or i >= self._rows:
                            neighbours.append(-1)
                            neighbours.append(-1)
                            neighbours.append(-1)
                            continue
                        for j in range(column-1, column+2):
                            if i == row and j == column:
                                continue
                            if j < 0 or j >= self._columns:
                                neighbours.append(-1)
                                continue

                            n_idx = i * self._columns + j
                            neighbours.append(n_idx)
                            if n_idx in mine_indices:
                                neighbouring_mines += 1

                    # tiles
                    is_mine = (row * self._columns + column) in mine_indices
                    state = 9  # unrevealed
                    # tile = Tile(neighbours, neighbouring_mines, is_mine)
                    self._board.append(
                        [state, neighbours, neighbouring_mines, is_mine])

        # create state
        return self.State()

    def render(self, mode="human", close=False):
        for row in range(self._rows):
            for column in range(self._columns):
                tile = self._board[row * self._columns + column]
                if tile[STATE] < 10:
                    print(tile[STATE], "", end=" ")
                else:
                    print(tile[STATE], end=" ")
            print("")

        print("")

    def State(self):
        states = []
        neighbours = []
        for t in self._board:
            states.append(t[STATE])
            # neighbours.append(t[NEIGHBOURS])
        state_np = np.array(states, dtype=np.float32)

        # convolution
        state_np = np.reshape(state_np, (-1, self._columns))
        state_np = np.expand_dims(state_np, axis=0)

        # print(state)

        return state_np

    def GetAction(self, row, column):
        return row * self._columns + column

    def Play(self):
        self.reset()
        self.render()

        while True:
            i, j = input("choose tile [row column] - rows: " + str(self._rows-1) + " columns: " + str(self._columns-1) + "\n>").split()
            action = self.GetAction(int(i), int(j))
            state, reward, done, win = self.step(action)
            print(reward, done, win)
            self.render()

            if done:
                if win:
                    print("You won!")
                else:
                    print("You lose!")

                _ = input("press enter to restart")
                self.reset()
                self.render()


    
