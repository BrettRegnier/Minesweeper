import random
import gym
import numpy as np 
import math
from gym import spaces

import pygame
import Globals
from Grid import Grid
from Menu import Menu 
from Screenshot import Screenshot

from Models.tnr.TNR import NumberClassifier

class Minesweeper(gym.Env):
    def __init__(self):
        super(Minesweeper, self).__init__()
        self.Init()
        self.InitGame()        
        
    def step(self, action):
        pass
        
    def reset(self):
        pass
        
    def render(self, mode="human", close=False):
        pass
        
    # Game logic below
    def Init(self):        
        # The display
        self._screen = None
        
        # Game clock
        self._clock = None
        
        # The width and height of the game window
        self._windowWidth = 424
        self._windowHeight = 444
        
        # Mouse click types
        self._mtype = 0
        
        # The Sweeping grid
        self._grid = None
        
        # Menu on the screen
        self._menu = None
        
        # Condition of the game running
        self._running = None
        
        # Frames per second
        self._fps = 60
        
        # The Ai used to predict what is in each square
        self._CNN = None
        
        ### Training for the CNN Variables ###
        # Used for determining if the the program should cycle fonts and screenshot
        self._screenshotFonts = False
        # Index of the current font in a list of fonts
        self._fontidx = 0
        # Condition if the screenshots letters should have random colours
        self._screenshotRandomColours = False
        # I don't know right now
        self._colourcount = 0
        
    def InitGame(self):
        print("Initializing Minesweeper")
        pygame.init()
        # Logo
        #logo = pygame.image.load("logo.png")
        #pygame.display.set_icon(logo)
        pygame.display.set_caption("Minesweeper")
        
        self._screen = pygame.display.set_mode([self._windowWidth, self._windowHeight])
        self._screen.fill([40, 40, 40])
        
        self._clock = pygame.time.Clock()
        
        print("Initializing CNN")
        self._CNN = NumberClassifier()
        
        print("Initializing Globals")
        pygame.font.init()
        SetDefaultFont()
        Globals._screenshot = Screenshot(self._screen)
        
        m = 20 # Margin
        # Box inside of window
        w = int(self._windowWidth - m*2) 
        h = int(self._windowHeight - m*2)
        self._grid = Grid(m, m*2, w, h, 2, self._screen)
        self._menu = Menu(self._windowWidth, self._windowHeight, self._screen)
        Globals._gameover = False
        
        print("Ready to play")
        
    def SetDefaultFont():
        Globals._fontname = "times new roman"
        Globals._font = pygame.font.SysFont(Globals._fontname, 24)    