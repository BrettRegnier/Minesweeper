import pygame
import Globals

class MenuItem:
    def __init__(self, string, x, y, subItems = None):
        self._string = string
        self._x = x
        self._y = y
        self._items = subItems #list of other menuItems
        
        # for checking on where the user clicked
        self._w, self_h = Globals.font.size(self._string)
        self._w = self._w + 6
        
    def Draw(self):
        surface = pygame.display.get_surface()
        textSurface = Globals.font.render(self._string, False, (0, 0, 0))
        surface.blit(textSurface, (self._x + 3, self._y))
    
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