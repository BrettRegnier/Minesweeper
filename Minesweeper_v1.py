import gym
import numpy as np
from gym import spaces

import pygame
from Board import Board
from Menu import Menu
from Cursor import Cursor
import State


class Minesweeper_v1(gym.Env):
    def __init__(self, human, difficulty):
        super(Minesweeper_v1, self).__init__()
        self._human = human  # true or false
        self.Init()
        self.InitGame(difficulty)

        self._actions = 0

        # AI variables
        self._steps = 0
        self._first_click = True

    def step(self, action):
        self._steps += 1
        x = -1
        y = -1
        was_revealed = False
        reward = 0
        done = False
        win = False
        tile_size = 32 # tile size of this board


        x = tile_size//2 + (action % self._columns) * tile_size
        # 20 is the margin from the top.
        y = 20 + tile_size//2 + action//self._columns * tile_size

        # move the cursor if the graphics are on
        self._cursor.SetPosition(x, y)

        was_revealed = self._board.Click(x, y, 0)

        #reward
        reward = -0.3
        if was_revealed:
            reward = 0.9

        # reward = 0
        
        done = False
        if State._gameover and State._win:
            reward = 1
            done = True
            win = True
            print("****win**** - ", end="")
        elif State._gameover: # or self._steps == (self._columns * self._rows) - self._mines:
            reward = -1
            done = True
            print("lose ", end="")

        # unrevealed = 10
        # revealed = 11
        # mine = 12
        # tile number = ##
        # get state after update.
        self.Update()
        state = self.State()

        if self._first_click:
            self._first_click = False
            reward = 0
        
        # print(state)
        # print("action:", action, "x:", x, "y:", y, "reward", reward, "done", done)

        return state, reward, done, win

    def reset(self):
        self.Restart(False)
        self._steps = 0
        self._first_click = True
        return self.State()

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
        if self._human:
            self._clock = pygame.time.Clock()
        self._fps = 60
        assert difficulty > 0
        size = 32
        rows = int(6 * difficulty + 2)
        columns = int(11 * difficulty - difficulty**2)
        mines = int(((0.006895 * difficulty**2) + 0.013045 * difficulty + 0.10506) * (rows * columns))
        mHeight = 20

        self._columns = columns
        self._rows = rows
        self._mines = mines

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
        self.action_space = spaces.Discrete(total)
        self.observation_space = spaces.Box(
            low=0, high=12, shape=(1, rows, columns), dtype=np.float32)
        # self.observation_space = spaces.Box(
        #     low=0, high=12, shape=(1, total), dtype=np.float32)

        print(self.action_space)
        print(self.observation_space)

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
        tick = 0
        if self._human:
            tick = self._clock.tick(self._fps)

        self._menu.Update(tick)
        self._board.Update(tick)
        self._cursor.Update(tick)

        if State._gameover:
            self.Gameover()
        if State._restart:
            self.reset()

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
        action = self._board.GetTileHover(x, y)
        if action != -1 and mtype == 0:
            self.step(action)
        else:
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

    def State(self):
        # tiles_2d = np.reshape(self._board._tiles, (-1, self._columns))
        state = []
        for tile in self._board._tiles:
            state.append(tile.GetState())
            
        state_np = np.array(state, dtype=np.float32)
        state_np = np.reshape(state_np, (-1, self._columns))
        state_np = np.expand_dims(state_np, axis=0)
        # print(state_np)
        # exit()
        return state_np

    def IsWin(self):
        return self._win

    def Gameover(self):
        if self._human == False:
            # if its a bot just restart
            State._restart = True
