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

import time


class Minesweeper_v0(gym.Env):
    def __init__(self):
        super(Minesweeper_v0, self).__init__()
        self.Init()
        self.InitGame()

        self._steps = 0

        self._observationSpace = spaces.Box(
            low=0, high=9, shape=(1, 144), dtype=np.int32)
        self._actionSpace = spaces.Discrete(144)
        
        self._wins = 0
        self._prevwins = 0
        self._loses = 0

    def step(self, action):
        self._steps += 1

        # Make a decision
        # quick math, this will allow for both the version that can see the game data and the one that cannot.
        size = 32
        xmargin = 20
        ymargin = 40

        x = xmargin + size/2 + (action % 12) * size
        y = ymargin + size/2 + int(action / 12) * size
        # time.sleep(2)

        # pygame.mouse.set_pos(x, y)
        revealed = self._grid.Click(x, y, 0)

        # update the game after the decision.
        self.Update()

        # reward
        reward = -1
        if revealed:
            reward = 1
        
        done = False
        if Globals._gameover == True and Globals._win == True:
            reward = 10
            done = True
            self._wins += 1
        elif Globals._gameover == True:
            reward = -10
            done = True
            self._loses += 1

        # unrevealed = -1
        # revealed & empty = 0
        # bomb = -2
        # tile number = ##
        # get state after action
        state = self._state

        # print(action, "x:", x, "y:", y, "reward", reward, "done", done)

        return state, reward, done, {}

    def reset(self):
        if self._wins % 10 == 0 and self._wins > self._prevwins:
            self.ResetGame(True)
            self._prevwins = self._wins
        else:            
            self.ResetGame(False)
        print ("loses:", self._loses, "wins:", self._wins) 

        return self._state

    def render(self, mode="human", close=False):
        if (mode == "human"):
            self.Render()

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
        self._quit = None

    def InitGame(self):
        print("Initializing Minesweeper")
        # pygame.init()
        # Logo
        #logo = pygame.image.load("logo.png")
        # pygame.display.set_icon(logo)
        pygame.display.set_caption("Minesweeper")

        self._screen = pygame.display.set_mode(
            [self._windowWidth, self._windowHeight])
        self._screen.fill([240, 240, 240])

        self._clock = pygame.time.Clock()

        # The cnn to play the game should not be part of the gym environment
        # because they are different tasks
        # print("Initializing CNN")
        # self._CNN = NumberClassifier()

        print("Initializing Globals")
        pygame.font.init()
        self.SetDefaultFont()
        Globals._screenshot = Screenshot(self._screen)

        m = 20  # Margin
        # Box inside of window
        w = int(self._windowWidth - m*2)
        h = int(self._windowHeight - m*2)
        self._grid = Grid(m, m*2, w, h, 1, self._screen)
        self._menu = Menu(self._windowWidth, self._windowHeight, self._screen)
        self._state = self._grid._state
        
        Globals._gameover = False
        Globals._win = False

        print("Ready to play")
    
    def ResetGame(self, hard):
        self._grid.Reset(hard)
        self._state = self._grid._state       
        
        Globals._gameover = False
        Globals._win = False

    def SetDefaultFont(self):
        Globals._fontname = "times new roman"
        Globals._font = pygame.font.SysFont(Globals._fontname, 24)

    # Begin game logic #

    # Display the game
    def Render(self):
        self._screen.fill((240, 240, 240))

        self.Draw()
        self.MouseHover()
        self.Events()

        pygame.display.update()

    def Update(self):        
        tick = self._clock.tick(self._fps)
        self._menu.Update(tick)
        self._grid.Update(tick)

        self._state = self._grid._state

    def Draw(self):
        self._menu.Draw()
        self._grid.Draw()

        if (Globals._gameover):
            self.Gameover()

    def MouseHover(self):
        x, y = pygame.mouse.get_pos()
        self._menu.MouseHover(x, y)

    def Events(self):
        # event handling
        for event in pygame.event.get():
            # quit
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (pygame.mouse.get_pressed()[0]):
                    self._mtype = 0  # left click
                elif (pygame.mouse.get_pressed()[2]):
                    self._mtype = 1  # right click
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                self._menu.Click(x, y, 0)

                if (Globals._gameover == False):
                    # Only click if not game over
                    self._grid.Click(x, y, self._mtype)
                    print(x, y)
            elif event.type == pygame.KEYUP:
                if Globals._MakeTrainingData == False:
                    if event.key == pygame.K_g:
                        self.ScreenshotGrid()
                        self._model.Predict()
                    elif event.key == pygame.K_s:
                        Globals._screenshot.CaptureWindow()
                        Globals._screenshot.Save("_Window", "./imgs/")
                    elif event.key == pygame.K_l:
                        self._screenshotRandomColors = True
                else:
                    print("In training mode can't screenshot normally.")
                    if event.key == pygame.K_p:
                        self.GetFontList()
                        self._screenshotFonts = True
                        # FPS = 1

                if event.key == pygame.K_t:
                    # Training mode.
                    Globals._MakeTrainingData = not Globals._MakeTrainingData
                    Globals._newgame = True

                    if Globals._MakeTrainingData:
                        print("Training Mode enabled")
                    else:
                        print("Training Mode disabled")
                

    def Gameover(self):
        # display gameover
        if Globals._win == False:
            textSurface = Globals._font.render(str("Game Over."), False, (0, 0, 0))
        elif Globals._win == True:
            textSurface = Globals._font.render(str("Victory!"), False, (0, 0, 0))
