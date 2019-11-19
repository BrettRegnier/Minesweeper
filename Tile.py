class Tile():
	def __init__(self, x, y, size, tid):
		self._x = x
		self._y = y
		self._size = size
		self._id = tid
	
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