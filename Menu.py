import pygame
import Globals
import MenuItem

class Menu:
    def __init__(self, wWidth, wHeight, screen):
        self._wW = wWidth
        self._wH = wHeight 
        self._screen = screen
        
        self._width = wWidth
        self._height = 20
        
        self._items = []
        newGame = MenuItem.MenuItem("New Game", 0, 0, screen)
        help = MenuItem.MenuItem("Help", newGame.Right(), 0, screen)
        
        self._items.append(newGame)
        self._items.append(help)
        
    def Update(self, tick):
        x = True
        
    def Draw(self):
        pygame.draw.line(self._screen, 
        (40, 40, 40), 
        (0,self._height), (self._width, self._height), 
        1)
        
        for item in self._items:
            if (item is not None):
                item.Draw()
        
    def MouseHover(self, mx, my):
        for item in self._items:
            if (item is not None):
                item.MouseHover(mx, my)
    
    def Click(self, mx, my, mType):
        for item in self._items:
            if (item is not None):
                item.Click(mx, my, mType)