class Cursor():
	def __init__(self, x, y, w, h):
		self._x = x
		self._y = y
		self._w = w
		self._h = h
	
	def Update(self, tick, mouse):
		self._x, self._y = mouse.get_pos()
	
	def Draw(self, screen, graphics):
		cx = self._x - self._w / 2
		cy = self._y - self._h / 2
		
		graphics.rect(screen, 
			(0, 0, 0),
			(cx, cy, self._w, self._h),
			2)
		
		# Horizontal line
		graphics.line(screen, 
			(0, 0, 0),
			(cx, self._y - 1),
			(self._x + self._w / 2, self._y - 1),
			2)
		
		# Vertical line
		graphics.line(screen, 
			(0, 0, 0),
			(self._x - 1, cy),
			(self._x - 1, self._y + self._h / 2),
			2)
		
	def MouseHover(self):
		return self.GetPosition()
	
	def Click(self):
		return self.GetPosition()
	
	def SetPosition(self, x, y):
		self._x = x
		self._y = y
	
	def GetPosition(self):
		return self._x, self._y
	