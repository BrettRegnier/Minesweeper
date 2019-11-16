import gym
import numpy as np
from gym import spaces

import pygame
import Globals
from Grid import Grid
from Menu import Menu
from Screenshot import Screenshot


class Minesweeper_v0(gym.Env):
    def __init__(self):
        super(Minesweeper_v0, self).__init__()
        # testing
        self._drawees = []
        self._updatees = []
        #end test
        self.Init()
        self.InitGlobals()
        self.InitGame(4)

        self._steps = 0

        self._wins = 0
        self._prevwins = 0
        self._loses = 0
        

    def step(self, action):
        self._steps += 1

        # Make a decision
        # quick math, this will allow for both the version that can see the
        # game data and the one that cannot.
        size = 32

        # if action == 2:
        #     print(action % self._columns)
        #     exit()

        x = size/2 + (action % self._columns) * size
        y = 20 + size/2 + int(action % self._rows) * size
        # import time 
        # time.sleep(2)

        # pygame.mouse.set_pos(x, y)
        revealed = self._grid.Click(x, y, 0)

        # update the game after the decision.
        self.Update()

        # reward
        reward = -0.3
        if revealed:
            reward = 0.9

        done = False
        if Globals._gameover and Globals._win:
            reward = 1
            done = True
            self._wins += 1
        elif Globals._gameover:
            reward = -1
            done = True
            self._loses += 1

        # unrevealed = -1
        # revealed & empty = 0
        # bomb = -2
        # tile number = ##
        # get state after action
        state = self._grid._state

        # print(action, "x:", x, "y:", y, "reward", reward, "done", done)

        return state, reward, done, {}

    def reset(self):
        if self._wins % 1 == 0 and self._wins > self._prevwins:
            self.ResetGame(True)
            self._prevwins = self._wins
        else:
            self.ResetGame(True)
        print(" losses:", self._loses, "wins:", self._wins)

        return self._grid._state

    def render(self, mode="human", close=False):
        if (mode == "human"):
            self.Render()

    # Game logic below
    def Init(self):
        # The display
        self._screen = None

        # Game clock
        self._clock = None

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

        # Training for the CNN Variables #
        # Used for determining if the program should cycle fonts and screenshot
        self._screenshotFonts = False
        # Index of the current font in a list of fonts
        self._fontidx = 0
        # Condition if the screenshots letters should have random colours
        self._screenshotRandomColours = False
        # I don't know right now
        self._colourcount = 0

    def InitGlobals(self):
        Globals._screenshot = Screenshot(self._screen)
        pygame.font.init()
        Globals._fontname = "times new roman"
        Globals._font = pygame.font.SysFont(Globals._fontname, 12)
        Globals._gameover = False
        Globals._win = False
        Globals._newgame = False
        Globals._colorsEnabled = True
        Globals._MakeTrainingData = False
        Globals._OverrideMineCount = 0
        Globals._TestCount = 0

    def InitGame(self, difficulty):
        self._clock = pygame.time.Clock()

        # The cnn to play the game should not be part of the gym environment
        # because they are different tasks
        # print("Initializing CNN")
        # self._CNN = NumberClassifier()

        size = 0
        mines = 0
        self._rows = 0
        self._columns = 0
        if difficulty == 0:
            mines = 10  # easy
            self._rows = 8
            self._columns = 10
            self._windowWidth = 320
            self._windowHeight = 276
        elif difficulty == 1:
            mines = 40  # medium
            self._rows = 14
            self._columns = 18
            self._windowWidth = 576
            self._windowHeight = 468
        elif difficulty == 2:
            mines = 99  # hard
            self._rows = 20
            self._columns = 24
            self._windowWidth = 768
            self._windowHeight = 660
        elif difficulty == 3:
            mines = 6  # very easy
            self._rows = 5
            self._columns = 7
            self._windowWidth = 224
            self._windowHeight = 180
        elif difficulty == 4:
            mines = 3  # extra very easy
            self._rows = 3
            self._columns = 4
            self._windowWidth = 128
            self._windowHeight = 116
        elif difficulty == 5:
            mines = 1  # ultra instinct easy
            self._rows = 2
            self._columns = 3
            self._windowWidth = 96
            self._windowHeight = 84

        total = self._rows * self._columns
        self._observationSpace = spaces.Box(
            low=0, high=9, shape=(1, total), dtype=np.int32)
        self._actionSpace = spaces.Discrete(total)

        self._grid = Grid(0, 20, self._windowWidth,
                          self._windowHeight, self._rows, self._columns, mines)
                          
        self._updatees.append(self._grid)
        self._drawees.append(self._grid)

    def InitGraphics(self):
        # Logo
        # logo = pygame.image.load("logo.png")
        # pygame.display.set_icon(logo)
        pygame.display.set_caption("Minesweeper")

        self._screen = pygame.display.set_mode(
            [self._windowWidth, self._windowHeight])
        self._screen.fill([240, 240, 240])

        self._grid.InitGraphics(self._screen)
        self._menu = Menu(self._windowWidth, self._windowHeight, self._screen)
        
        self._updatees.append(self._menu)
        self._drawees.append(self._menu)

        pygame.font.init()

    def ResetGame(self, hard):
        self._grid.Reset(hard)

        Globals._gameover = False
        Globals._win = False

    # Begin game logic #
    # Display the game
    def Render(self):
        if (self._screen == None):
            self.InitGraphics()

        self._screen.fill((240, 240, 240))

        self.Draw()
        self.MouseHover()
        self.Events()

        pygame.display.update()

    def Update(self):
        tick = self._clock.tick(self._fps)
        
        for up in self._updatees:
            up.Update(tick)

    def Draw(self):
        for drawee in self._drawees:
            drawee.Draw()

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
                    print(self._grid._state)

    def Play(self):
        self.InitGraphics()
        self._running = True
        while self._running:
            self.Update()
            self.Render()

    def Gameover(self):
        # display gameover
        if Globals._win == False:
            textSurface = Globals._font.render(
                str("Game Over."), False, (0, 0, 0))
        elif Globals._win == True:
            textSurface = Globals._font.render(
                str("Victory!"), False, (0, 0, 0))
