import pygame
import Globals
import Button

class Menu:
    def __init__(self, wWidth, wHeight, screen):
        self._wW = wWidth
        self._wH = wHeight 
        self._screen = screen
        
        self._width = wWidth
        self._height = 20
        
        self._items = []
        
        def ngfunc():
            Globals._newgame = True
            
        btnNewGame = Button.Button("New Game", 0, 0, screen, ngfunc)
        
        
        btnHelp = Button.Button("Help", btnNewGame.Right(), 0, screen, ngfunc)
        
        self._items.append(btnNewGame)
        self._items.append(btnHelp)
        
    def Update(self, tick):
        pass
        
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