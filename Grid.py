import pygame
import random
import Tile


class Grid:
    def __init__(self, x, y, width, height, difficulty, screen):
        if (difficulty == 1):
            # easy
            self._mines = 10
        elif (difficulty == 2):
            # med
            self._mines = 25
        elif (difficulty == 3):
            # hard
            self._mines = 40

        self._x = x
        self._y = y
        self._width = width
        self._height = height
        
        self._screen = screen

        self.Build()

    def Build(self):
        # first tile
        x = 0
        y = 0
        size = 20
        self._tiles = []

        rows = int(self._height / size)
        columns = int(self._width / size)
        total = rows*columns
        count = 0

        # create tiles 1-d
        for r in range(rows):
            for c in range(columns):
                self._tiles.append(Tile.Tile(self._x + x, self._y + y, size, count, self._screen, total))
                x = x + size

                # For training
                # self._tiles[count].OverrideValue(count)

                count = count + 1

            x = 0
            y = y + size

        toMakeMine = self._mines
        # toMakeMine = 0
        # make them mine
        while (toMakeMine > 0):
            tile = random.choice(self._tiles)
            if (not tile.IsMine()):
                tile.BeMine()
                toMakeMine = toMakeMine - 1

        # make tiles known to eachother
        tpr = columns  # Tiles per row
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
        # animations?
        for tile in self._tiles:
            tile.Update(tick)

    def Draw(self):
        # draw tiles
        for tile in self._tiles:
            tile.Draw()

    def Click(self, mx, my, mtype):
        for tile in self._tiles:
            if (mx > self._x and mx < self._x + self._width and
                    my > self._y and my < self._y + self._height):
                tile.Click(mx, my, mtype)

    def Screenshot(self):
        for tile in self._tiles:
            tile.Screenshot()
            
    # training purposes
    def ChangeColors(self):
        for tile in self._tiles:
            tile.RandomizeColor()