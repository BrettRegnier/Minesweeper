

class Button:
    def __init__(self, string, font, x, y, w, h, func, subItems=None):
        self._string = string
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._func = func
        self._items = subItems  # list of other menuItems
        self._isHovering = False
        self._font = font

        # for checking on where the user clicked
        self._w, _ = self._font.size(self._string)
        self._w = self._w + 6

    def Draw(self, screen, graphics):
        if (self._isHovering):
            graphics.rect(screen,
                          (200, 200, 200),
                          (self._x, self._y,
                           self._x + self._w, self._y + self._h))

        textSurface = self._font.render(self._string, False, (0, 0, 0))
        screen.blit(textSurface, (self._x + 3, self._y + 2))

    def Update(self, tick):
        self._isHovering = False

    def MouseHover(self, mx, my):
        if (mx > self._x and mx < self._x + self._w and my > self._y and my < self._y + self._h):
            self._isHovering = True

    def Click(self, mx, my, mType):
        if (mx > self._x and mx < self._x + self._w and my > self._y and my < self._y + self._h and mType == 0):
            self._func()

    def Left(self):
        return self._x

    def Right(self):
        return self._x + self._w
