import pygame
import os

import Globals

class Screenshot:
    def __init__(self, screen):
        self._img = None
        self._screen = screen
        
    def Capture(self, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        self._img = self._screen.subsurface(rect)
    
    def CaptureWindow(self):
        self._img = self._screen
        
    def Save(self, name, path="./"):
        if (not os.path.exists(path)):
            os.makedirs(path)
        pygame.image.save(self._img, path + str(name) + ".jpg")
        
        if (not Globals._MakeTrainingData):
            print('saved tile ' + str(name))