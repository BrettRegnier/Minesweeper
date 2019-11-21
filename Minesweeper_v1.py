import gym
import numpy as np
from gym import spaces

import pygame
from Board import Board
from Menu import Menu
from Cursor import Cursor

class Minesweeper_v1(gym.Env):
	def __init__(self):
		super(Minesweeper_v1, self).__init__()
		self.Init()
		self.InitGame(1)

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

		self._wWidth = columns * 32
		self._wHeight = rows * 32 + mHeight
		
		self._cursor = Cursor(10, 10, 11, 11)
		
		self._menu = Menu(0, mHeight, self._wWidth, self._wHeight)

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
		pygame.mouse.set_cursor((8,8),(0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0))

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
		self._cursor.Update(tick, pygame.mouse)

	def Draw(self):
		for d in self._drawees:
			d.Draw(self._screen, self._graphics)

	def MouseHover(self):
		# for d in self._drawees:
		# 	d.MouseHover()
		pass
	
	# TODO event handles
	def Events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self._running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				press = pygame.mouse.get_pressed()
				if (press[0]):
					pass
				
	def Click(self, mtype):
		x, y = self._cursor.Click()
	# TODO cursor			

	def Restart(self, hard):
		pass

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
		pass
