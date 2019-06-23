import pygame

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
        pygame.image.save(self._img, path + str(name) + ".jpg")
        print('saved')