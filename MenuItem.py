import pygame
import Globals

class MenuItem:
    def __init__(self, string, x, y, screen, subItems = None):
        self._string = string
        self._x = x
        self._y = y
        self._items = subItems #list of other menuItems
        self._screen = screen
        
        # for checking on where the user clicked
        self._w, self_h = Globals._font.size(self._string)
        self._w = self._w + 6
        
    def Draw(self):
        textSurface = Globals._font.render(self._string, False, (0, 0, 0))
        self._screen.blit(textSurface, (self._x + 3, self._y))
    
    def Update(self, tick):
        x = True
    
    def MouseHover(self, mx, my):
        x = True
        
    def Click(self, mx, my, mType):
        x = True
        
    def Left(self):
        return self._x
    
    def Right(self):
        return self._x + self._w