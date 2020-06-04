import random

from Tile import Tile
import State


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

        self._tiles = []
        self._state = []

        # tiles that are not mines that are unrevealed should be updated in click
        self._unrevealed = 0
        self._tileSize = 32

        self.CreateBoard()

    def CreateBoard(self):
        x = 0
        y = 0
        total = self._rows * self._columns

        self._unrevealed = total - self._mines
        self._tiles = []
        self._state = []

        count = 0

        # create tiles 1-d
        for r in range(self._rows):
            for c in range(self._columns):
                tile = Tile(self._x + x, self._y + y, self._tileSize, count)
                self._tiles.append(tile)
                x = x + self._tileSize

                # set the state to be all unrevealed - 10
                self._state.append(tile.GetState())
                count = count + 1

            x = 0
            y = y + self._tileSize

        # Make mines
        toMakeMine = self._mines
        while (toMakeMine > 0):
            tile = random.choice(self._tiles)
            if (not tile.IsMine()):
                tile.BeMine()
                toMakeMine -= 1

        # Make tiles known to eachother
        tpr = self._columns  # Tiles per row
        count = len(self._tiles)
        for i in range(count):
            r = int(i / tpr)  # current row
            c = int(i % tpr)  # current column

            testidx = -1

            # adjacent tiles
            adj = []

            # Northwest tile
            NW = i - (tpr + 1)
            if (NW >= 0 and int(NW / tpr) == r - 1):
                adj.append(self._tiles[NW])

                if i == testidx:
                    print("NW" + str(NW))
            else:
                adj.append(None)

            # North tile
            N = i - tpr
            if (N >= 0):
                adj.append(self._tiles[N])
                if i == testidx:
                    print("N" + str(N))
            else:
                adj.append(None)

            # Northeast tile
            NE = i - (tpr - 1)
            if (NE >= 0 and int(NE / tpr) == r - 1):
                adj.append(self._tiles[NE])
                if i == testidx:
                    print("NE" + str(NE))
            else:
                adj.append(None)

            # West tile
            W = i - 1
            if (W > 0 and int(W / tpr) == r):
                adj.append(self._tiles[W])
                if i == testidx:
                    print("W" + str(W))
            else:
                adj.append(None)

            # East tile
            E = i + 1
            if (E < count and int(E / tpr) == r):
                adj.append(self._tiles[E])
                if i == testidx:
                    print("E" + str(E))
            else:
                adj.append(None)

            # Southwest tile
            SW = i + (tpr - 1)
            if (SW < count and int(SW / tpr) == r + 1):
                adj.append(self._tiles[SW])
                if i == testidx:
                    print("SW" + str(SW))
            else:
                adj.append(None)

            # South tile
            S = i + tpr
            if (S < count):
                adj.append(self._tiles[S])
                if i == testidx:
                    print("S" + str(S))
            else:
                adj.append(None)

            # Southeast tile
            SE = i + (tpr+1)
            if (SE < count and int(SE / tpr) == r + 1):
                adj.append(self._tiles[SE])
                if i == testidx:
                    print("SE" + str(SE))
            else:
                adj.append(None)

            self._tiles[i].AdjacentTiles(adj)

    def Update(self, tick):
        for tile in self._tiles:
            tile.Update(tick)
        if State._gameover:
            for tile in self._tiles:
                tile.RevealSelf()

    def Draw(self, screen, graphics):
        for tile in self._tiles:
            tile.Draw(screen, graphics)

    def MouseHover(self, mx, my):
        return
        idx = self.GetTileHover(mx, my)
        if idx >= 0:
            self._tiles[idx].MouseHover()

    def Click(self, mx, my, mtype):
        was_revealed = False
        is_mine = False # technically win condition should be handled outside of this class.
        if State._gameover == False:
            idx = self.GetTileHover(mx, my)
            if (idx >= 0):
                if mtype == 0:
                    queue = [self._tiles[idx]]
                    while len(queue) > 0:
                        tile = queue.pop(0)
                        revealed, is_mine = tile.Click(mtype)
                        if tile._nearbyMines == 0 and not tile.IsMine():
                            for adj in tile._adjTiles:
                                if adj != None and not adj.Revealed():
                                    queue.append(adj)
                        if revealed:
                            self._unrevealed -= 1
                            was_revealed = True
                else:
                    self._tiles[idx].Click(mtype)

                idx = 0
                for t in self._tiles:
                    self._state[idx] = t.GetState()
                    idx += 1

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
            for tile in self._tiles:
                tile.Reset()
