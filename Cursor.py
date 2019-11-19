class Cursor():
	def __init__(self, x, y, w, h):
		self._x = x
		self._y = y
		self._w = w
		self._h = h
	
	def Update(self, tick, mouse):
		self._x, self._y = mouse.get_pos()
	
	def Draw(self, screen, graphics):
		graphics.rect(screen, 
			(0, 0, 0),
			(self._x - self._w/2, self._y - self._h/2, self._w, self._h),
			2)
	
		# graphics.line(screen, 
		# 	(0, 0, 0),
		# 	(self._x, int((self._y+1)/2)),
		# 	(self._w, int(self._y+1)))
		
	def MouseHover(self):
		pass
	
	def Click(self):
		pass
	
	def SetPosition(self, x, y):
		self._x = x
		self._y = y
	
	def GetPosition(self):
		return self._x, self._y
	