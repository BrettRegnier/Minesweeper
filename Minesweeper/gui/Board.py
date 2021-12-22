import random

from Minesweeper.gui.Tile import Tile

import Minesweeper.utils.GameState

import numpy as np

# TODO make game state class that is global.


class Board():
    def __init__(self, x, y, w, h, rows, columns, mines):
        self._x = x
        self._y = y
        self._width = w
        self._height = h
        self._mines = mines

        self._rows = rows
        self._columns = columns

        # 1D array of tiles.
        self._tiles = None

        # tiles that are not mines that are unrevealed should be updated in click
        self._unrevealed = 0
        self._tileSize = 32

        self.CreateBoard()

    def CreateBoard(self):
        x = 0
        y = 0
        total = self._rows * self._columns
        self._unrevealed = total - self._mines
        self._tiles = np.empty((self._rows * self._columns), dtype=Tile)
        
        # Choose mines
        to_make_mine = self._mines
        mines = []
        choices = [c for c in range(total)]
        while (to_make_mine > 0):
            idx = random.choice(choices)
            mines.append(idx)
            choices.remove(idx) 
            to_make_mine -= 1

        count = 0
        for row in range(self._rows):
            for column in range(self._columns):

                # get neighbouring tiles
                neighbours = []
                neighbouring_mines = 0
                for i in range(row-1,row+2):
                    if i < 0 or i >= self._rows:
                        neighbours.append(None)
                        neighbours.append(None)
                        neighbours.append(None)
                        continue
                    for j in range(column-1, column+2):
                        if i == row and j == column:
                            continue
                        if j < 0 or j >= self._columns:
                            neighbours.append(None)
                            continue
                        
                        n_idx = i * self._columns + j
                        neighbours.append(n_idx)
                        if n_idx in mines:
                            neighbouring_mines += 1
                
                # create the tile
                tile = Tile(self._x + x, self._y + y, self._tileSize, count, neighbours, neighbouring_mines)
                idx = row * self._columns + column
                self._tiles[idx] = tile
                if idx in mines:
                    self._tiles[idx].BeMine()
                
                # shift over the postion
                x = x + self._tileSize
                count = count + 1
        
            # reset and shift down
            x = 0
            y = y + self._tileSize

    def Update(self, tick):
        for tile in self._tiles:
            tile.Update(tick)
        # if State._gameover:
        #     for tile in self._tiles:
        #         tile.Reveal()

    def Draw(self, screen, graphics):
        for tile in self._tiles:
            tile.Draw(screen, graphics)

    def MouseHover(self, mx, my):
        idx = self.GetTileHover(mx, my)
        if idx >= 0:
            self._tiles[idx].MouseHover()

    def Click(self, mx, my, mtype):
        was_revealed = False
        # technically win condition should be handled outside of this class.
        is_mine = False
        if State._gameover == False:
            idx = self.GetTileHover(mx, my)
            if (idx >= 0):
                if mtype == 0:
                    queue = [self._tiles[idx]]
                    while len(queue) > 0:
                        tile = queue.pop(0)
                        revealed, is_mine = tile.Click(mtype)
                        if tile._neighbouring_mines == 0 and not tile.IsMine():
                            for n_idx in tile._neighbours:
                                if n_idx != None:
                                    adj = self._tiles[n_idx]
                                    if not adj.Revealed():
                                        queue.append(adj)
                        if revealed:
                            self._unrevealed -= 1
                            # print(self._unrevealed)
                            was_revealed = True
                else:
                    self._tiles[idx].Click(mtype)

                # game over condition
                if is_mine:
                    State._gameover = True  # if it hits a mine then it will be true
                elif self._unrevealed == 0:
                    State._win = True
                    State._gameover = True

                # print("Tile idx:", idx)
                # print("Tile State:", tile.GetState())
                # print("Tile was revealed?", was_revealed)
                # print("Tile was mine?", wasMine)

        return was_revealed

    def GetTileHover(self, mx, my):
        # cut down on computational time
        # rows * columns + index
        if (mx >= self._x and mx <= self._x + self._width and my >= self._y and my <= self._y + self._height):
            column = (mx - self._x) // self._tileSize
            row = (my - self._y) // self._tileSize
            # print("column: ", column, "row: ", row)
            idx = row * self._columns + column
            return int(idx)
        return -1

    def Reset(self, hard):
        State._gameover = False

        if hard:
            self.CreateBoard()
        else:
            self._unrevealed = self._columns * self._rows - self._mines
            for tile in self._tiles:
                tile.Reset()
