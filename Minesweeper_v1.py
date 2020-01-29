import gym
import numpy as np
from gym import spaces

import pygame
from Board import Board
from Menu import Menu
from Cursor import Cursor
import State


class Minesweeper_v1(gym.Env):
    def __init__(self, human):
        super(Minesweeper_v1, self).__init__()
        self._human = human  # true or false
        self.Init()
        self.InitGame(0)

        self._actions = 0

        # AI variables
        self._steps = 0
        self._wins = 0
        self._losses = 0
        

    def step(self, action):
        self._steps += 1
        x = -1
        y = -1
        wasRevealed = False
        reward = 0
        done = False
        tilesize = 32 # tile size of this board


        x = tilesize/2 + (action % self._columns) * tilesize
        # 20 is the margin from the top.
        y = 20 + tilesize/2 + int(action % self._rows) * tilesize

        # move the cursor if the graphics are on
        self._cursor.SetPosition(x, y)
        wasRevealed = self._board.Click(x, y, 0)

        # update after click
        self.Update()

        #reward
        reward = -0.3
        if wasRevealed:
            reward = 0.9
        
        done = False
        if State._gameover and State._win:
            reward = 1
            done = True
            self._wins += 1
        elif State._gameover:
            reward = -1
            done = True
            self._losses += 1

        # unrevealed = 10
        # revealed = 11
        # mine = 12
        # tile number = ##
        # get state after update.
        state = self._board._state

        # print(action, "x:", x, "y:", y, "reward", reward, "done", done)

        return state, reward, done, {}

    def reset(self):
        self.Restart(True)
        print("losses:", self._losses, "wins:", self._wins)

        return self._board._state

    def render(self, mode="human", close=False):
        if (mode == "human"):
            if self._display is None:
                self.InitGraphics()
            self.Render()

    def Init(self):
        self._wWidth = 0
        self._wHeight = 0

        self._display = None
        self._screen = None
        self._graphics = None
        self._clock = None

        self._board = None
        self._cursor = None
        self._menu = None

        self._running = False
        self._fps = None
        self._font = None

        self._drawees = []

    def InitGame(self, difficulty):
        self._clock = pygame.time.Clock()
        self._fps = 60

        size = 32
        mines = 0
        rows = 0
        columns = 0
        mHeight = 20
        if difficulty == 0:
            mines = 10  # easy
            rows = 8
            columns = 10
        elif difficulty == 1:
            mines = 40  # medium
            rows = 14
            columns = 18
        elif difficulty == 2:
            mines = 99  # hard
            rows = 20
            columns = 24
        elif difficulty == 3:
            mines = 6  # very easy
            rows = 5
            columns = 7
        elif difficulty == 4:
            mines = 3  # extra very easy
            rows = 3
            columns = 4
        elif difficulty == 5:
            mines = 1  # ultra instinct easy
            rows = 2
            columns = 3

        self._columns = columns
        self._rows = rows

        self._wWidth = columns * 32
        self._wHeight = rows * 32 + mHeight

        mouse = None
        if (self._human):
            mouse = pygame.mouse
        self._cursor = Cursor(16, 36, 11, 11, mouse)

        self._menu = Menu(0, 0, self._wWidth, mHeight)

        self._board = Board(0, mHeight,
                            self._wWidth, self._wHeight,
                            rows, columns,
                            mines)

        total = rows * columns
        self._observationSpace = spaces.Box(
            low=0, high=9, shape=(1, total), dtype=np.int32)
        self._actionSpace = spaces.Discrete(total)

    def InitGraphics(self):
        pygame.display.init()
        pygame.display.set_caption("Minesweeper")
        pygame.font.init()
        pygame.font.SysFont("Times New Roman", 12)
        pygame.mouse.set_cursor(
            (8, 8), (0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0))

        self._display = pygame.display
        self._screen = self._display.set_mode(
            [self._wWidth, self._wHeight])
        self._screen.fill([240, 240, 240])

        self._graphics = pygame.draw

        self._drawees.append(self._menu)
        self._drawees.append(self._board)
        self._drawees.append(self._cursor)

    def Update(self):
        tick = self._clock.tick(self._fps)

        self._menu.Update(tick)
        self._board.Update(tick)
        self._cursor.Update(tick)

        if State._gameover:
            self.Gameover()
        if State._restart:
            self.Restart(True)

    def Draw(self):
        for d in self._drawees:
            d.Draw(self._screen, self._graphics)

        # gameover
        if State._gameover:
            msg = "Gameover!"
            if State._win:
                msg = "Victory!"
            pygame.font.init()

            font = pygame.font.SysFont("Times New Roman", 42)
            shadow = font.render(msg, True, (0, 0, 0))
            text = font.render(msg, True, (240, 240, 240))
            ow, oh = font.size(msg)
            px = self._wWidth/2 - ow/2
            py = (self._wHeight/2 - oh/2) - 20

            # shadow
            self._screen.blit(shadow, (px + 2, py))

            # acutal text
            self._screen.blit(text, (px, py))

    def MouseHover(self):
        x, y = self._cursor.Click()
        self._menu.MouseHover(x, y)
        self._board.MouseHover(x, y)

    # TODO event handles
    def Events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                press = pygame.mouse.get_pressed()
                if (press[0]):
                    self.Click(0)
                elif (press[2]):
                    self.Click(1)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_h:
                    self._cursor.SetPosition(10, 20)

    def Click(self, mtype):
        x, y = self._cursor.Click()
        self._board.Click(x, y, mtype)
        self._menu.Click(x, y, mtype)

    def Restart(self, hard):
        State._restart = False
        State._gameover = False
        State._win = False
        self._board.Reset(hard)

    def Play(self):
        self.InitGraphics()
        self._running = True
        while self._running:
            self.Update()
            self.Render()

    def Render(self):
        self._screen.fill((240, 240, 240))

        self.Events()
        self.MouseHover()
        self.Draw()

        self._display.update()

    def Gameover(self):
        if self._human == False:
            # if its a bot just restart
            State._restart = True
