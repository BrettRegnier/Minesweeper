from Tile import Tile

class Board():
	def __init__(self, x, y, w, h, rows, columns, mines):
		self._x = x
		self._y = y
		self._width = width
		self._height = height
		self._mines = mines
		
		self._rows = rows
		self._columns = columns
		
		self._tiles = []
		self._state = []
		
		self._unrevealed = 0

	def CreateBoard(self):
		x = 0
		y = 0
		size = 32
		total = self._rows * self._columns
		count = 0
		
		# create tiles 1-d
		for r in range(self._rows):
			for c in range(self._columns):
				self._tiles.append(Tile(self._x + x, self._y + y, size))
				x = x + size
				
				self._state.append(10) # set the state to be all unrevealed - 10
				count = count + 1

			x = 0
			y = y + size

	# might not be needed
	def CreateTile(self):
		pass

	def Update(self, tick):
		pass

	def Draw(self, screen, graphics):
		pass

	def MouseHover(self, mx, my):
		pass

	def Click(self, mx, my):
		pass

	def Reset(self, hard):
		pass
