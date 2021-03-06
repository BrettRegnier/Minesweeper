from pygame import font


class Tile():
    def __init__(self, x, y, size, tid, neighbours, neighbouring_mines):
        self._x = x
        self._y = y
        self._size = size
        self._id = tid

        self._mine = False
        self._neighbours = neighbours
        self._neighbouring_mines = neighbouring_mines

        self._revealed = False
        self._flagged = False
        self._state = 10

        self._outercolor = 255
        self._innercolor = 200
        self._revealedcolor = 180
        self._outline = 100
        self._textcolor = (0, 0, 0)
        self.SetTextColourOnMines()

        self._hover = False

        font.init()
        self._font = font.SysFont("Times New Roman", 12)

    def Update(self, tick):
        self._hover = False

    def Draw(self, screen, graphics):
        # border
        graphics.rect(screen,
                      (self._outline, self._outline, self._outline),
                      (self._x, self._y, self._size, self._size))

        # revealed tile
        if (self._revealed):
            c = self._revealedcolor
            nx = self._x + 1
            ny = self._y + 1
            ns = self._size - 2

            graphics.rect(screen,
                          (c, c, c),
                          (nx, ny, ns, ns))

            if (self._mine):
                # has a mine

                # circle
                graphics.ellipse(screen,
                                 (0, 0, 0),
                                 (nx+7, ny+7,
                                     ns-15, ns-15))

                # cross
                lstart = (self._x + 6, self._y + 6)
                lend = (self._x + self._size - 8, self._y + self._size - 8)
                graphics.line(screen,
                              (0, 0, 0),
                              lstart,
                              lend,
                              3)

                graphics.line(screen,
                              (0, 0, 0),
                              (lend[0], lstart[1]),
                              (lstart[0], lend[1]),
                              3)

            elif (self._neighbouring_mines > 0):
                # mines nearby
                textSurface = self._font.render(str(self._neighbouring_mines),
                                                False,
                                                self._textcolor)

                screen.blit(textSurface, (self._x + 10, self._y + 5))
        # unrevealed tile
        else:
            # decor
            ci = self._innercolor
            co = self._outercolor

            # outer design
            nx = self._x + 1
            ny = self._y + 1
            ns = self._size - 2

            # if hovering do a different outline maybe make it prettier
            if self._hover:
                nx += 1
                ny += 1
                ns -= 1

            graphics.rect(screen,
                          (co, co, co),
                          (nx, ny, ns, ns))

            # inner design
            nx = self._x + 3
            ny = self._y + 3
            ns = self._size - 4
            graphics.rect(screen,
                          (ci, ci, ci),
                          (nx, ny, ns, ns))

            if (self._flagged):
                cx = int(self._x + (self._size / 2))
                cy = int(self._y + (self._size / 2))
                tip = int(self._size / 2 - 4)

                graphics.polygon(screen,
                                 (255, 0, 0),
                                 [[cx, ny+2], [cx, cy+2], [cx+tip, cy]])

                graphics.line(screen,
                              (255, 0, 0),
                              [cx, cy], [cx, self._y + self._size - 3])

    def MouseHover(self):
        self._hover = True

    def Click(self, mtype):
        was_revealed = False
        if (self._revealed == False):
            if (mtype == 0):
                self.Reveal()
                was_revealed = True
            elif (mtype == 1):
                self.Flag()
        return was_revealed, self.IsMine()

    def Reset(self):
        self._flagged = False
        self._revealed = False
        self.SetState()

    def Reveal(self):
        self._revealed = True
        self.SetState()

    def Revealed(self):
        return self._revealed

    def Flag(self):
        self._flagged = not self._flagged

    def IsMine(self):
        return self._mine

    def BeMine(self):
        # state = 11
        self._mine = True

    def GetState(self):
        return self._state

    def SetState(self):
        if self._revealed:
            if self._mine:
                self._state = 12  # revealed mine
            elif self._neighbouring_mines == 0:
                self._state = 11 # revealed tile
            else:    
                self._state = self._neighbouring_mines
        else:
            self._state = 10  # unrevealed

    def SetTextColourOnMines(self):
        if (self._mine == False):
            # skipping == 1 because its default color.
            if self._neighbouring_mines == 2:
                self._textcolor = (71, 92, 68)
            elif self._neighbouring_mines == 3:
                self._textcolor = (226, 78, 27)
            elif self._neighbouring_mines == 4:
                self._textcolor = (135, 61, 72)
            elif self._neighbouring_mines == 5:
                self._textcolor = (25, 12, 14)
            elif self._neighbouring_mines == 6:
                self._textcolor = (63, 48, 71)
            elif self._neighbouring_mines == 7:
                self._textcolor = (140, 75, 92)
            elif self._neighbouring_mines == 8:
                self._textcolor = (53, 129, 184)
            
