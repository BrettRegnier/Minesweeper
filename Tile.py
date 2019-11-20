class Tile():
	def __init__(self, x, y, size, tid):
		self._x = x
		self._y = y
		self._size = size
		self._id = tid
		
		self._mine = False
		self._adjTiles = []
		
		self._revealed = False
		self._flagged = False
		self._state = -10
		
		self._outercolor = 255
		self._innercolor = 200        
		self._revealedcolor = 180
		self._outline = 100
		self._textcolor = (0, 0, 0)
	
	def Update(self, tick):
		pass
	
	def Draw(self, screen, graphics):
		pass
	
	def MouseHover(self, mx, my):
		pass
	
	def Click(self, mx, my):
		pass
	
	def Reset(self):
		pass
	
	def IsMine(self):
		return self._isMine
		
	def BeMine(self):
		self._state = 11
		self._mine = True
		
	def GetState(self):
		return self._state
		
	def AdjacentTiles(self, adj):
		self._adjTiles = adj
		
		for tile in self._adjacentTiles:
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
