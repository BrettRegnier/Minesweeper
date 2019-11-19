import pygame
import Globals

class Button:
	def __init__(self, string, x, y, screen, func, subItems = None):
		self._string = string
		self._x = x
		self._y = y
		self._items = subItems #list of other menuItems
		self._screen = screen
		self._isHovering = False
		self._func = func
		
		# for checking on where the user clicked
		self._w, self._h = Globals._font.size(self._string)
		self._w = self._w + 6
		
	def Draw(self):
		if (self._isHovering):            
			pygame.draw.rect(self._screen, 
			(200, 200, 200), 
			(self._x, self._y, self._w, self._h))
			
		textSurface = Globals._font.render(self._string, False, (0, 0, 0))
		self._screen.blit(textSurface, (self._x + 3, self._y))
	
	def Update(self, tick):
		pass
	
	def MouseHover(self, mx, my):
		if (mx > self._x and mx < self._x + self._w and my > self._y and my < self._y + self._h):
			self._isHovering = True
		else:
			self._isHovering = False
		
	def Click(self, mx, my, mType):
		if (mx > self._x and mx < self._x + self._w and my > self._y and my < self._y + self._h and mType == 0):
			self._func()
		
	def Left(self):
		return self._x
	
	def Right(self):
		return self._x + self._w