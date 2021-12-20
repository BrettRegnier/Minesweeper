from pygame import font

import minesweeper.gui.State as State
from minesweeper.gui.Button import Button


class Menu():
    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

        self._items = []

        font.init()
        self._font = font.SysFont("Times New Roman", 12)

        # New game button
        def ngfunc():
            State._restart = True

        btnWidth = 50
        btnHeight = self._h

        btnNewGame = Button("New Game", self._font, 0, 0,
                            btnWidth, btnHeight, ngfunc)

        self._items.append(btnNewGame)

    def Update(self, tick):
        for item in self._items:
            item.Update(tick)

    def Draw(self, screen, graphics):
        graphics.line(screen,
                      (40, 40, 40),
                      (0, self._h), (self._w, self._h),
                      1)

        for item in self._items:
            item.Draw(screen, graphics)

    def MouseHover(self, mx, my):
        if self.MouseInBounds(mx, my):
            for item in self._items:
                item.MouseHover(mx, my)

    def Click(self, mx, my, mType):
        if self.MouseInBounds(mx, my):
            for item in self._items:
                item.Click(mx, my, mType)

    def MouseInBounds(self, mx, my):
        return (mx >= self._x and mx <= self._x + self._w and
                my >= self._y and my <= self._y + self._h)
