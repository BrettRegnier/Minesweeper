from pygame import font

class Tile():
	def __init__(self, x, y, size, tid):
		self._x = x
		self._y = y
		self._size = size
		self._id = tid
		
		self._mine = False
		self._nearbyMines = 0
		self._adjTiles = []
		
		self._revealed = False
		self._flagged = False
		self._state = -10
		
		self._outercolor = 255
		self._innercolor = 200        
		self._revealedcolor = 180
		self._outline = 100
		self._textcolor = (0, 0, 0)
		
		font.init()
		self._font = font.SysFont("Times New Roman", 12)
	
	def Update(self, tick):
		pass
	
	def Draw(self, screen, graphics):
		#border
		graphics.rect(screen, 
			(self._outline, self._outline, self._outline), 
			(self._x, self._y, self._size, self._size))
			
		# revealed tile
		if (self._revealed):
			c = self._revealedcolor
			nx = self._x + 1
			ny = self._y + 1
			ns = self._size -2
			
			graphics.rect(screen, 
				(c, c, c), 
				(nx, ny, ns, ns))
					
			if (self._mine):
				# has a mine
				graphics.ellipse(screen,
					(0, 0, 0), 
					(nx+2, ny+2, 
					ns-4, ns-4))
					
			elif (self._nearbyMines > 0):
				# mines nearby
				textSurface = self._font.render(str(self._nearbyMines),
					False, 
					self._textcolor)
					
				screen.blit(textSurface, (self._x + 10, self._y + 5))
	
	def MouseHover(self):
		pass
	
	def Click(self):
		pass
	
	def Reset(self):
		pass
	
	def IsMine(self):
		return self._mine
		
	def BeMine(self):
		# state = 11
		self._mine = True
		
	def GetState(self):
		return self._state
		
	def AdjacentTiles(self, adj):
		self._adjTiles = adj
		
		for tile in self._adjTiles:
			if (tile is not None and tile.IsMine()):
				self._nearbyMines = self._nearbyMines + 1
				
		# skipping == 1 because its default color.
		if self._nearbyMines == 2:
			self._textcolor = (71, 92, 68)
		elif self._nearbyMines == 3:
			self._textcolor = (226, 78, 27)
		elif self._nearbyMines == 4:
			self._textcolor = (135, 61, 72)
		elif self._nearbyMines == 5:
			self._textcolor = (25, 12, 14)
		elif self._nearbyMines == 6:
			self._textcolor = (63, 48, 71)
		elif self._nearbyMines == 7:
			self._textcolor = (140, 75, 92)        
		elif self._nearbyMines == 8:
			self._textcolor = (53, 129, 184)
