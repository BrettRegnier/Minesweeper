import pygame
import Grid

global win #window
global clock
global running
global mtype
global font

global windowWidth
global windowHeight

global grid

windowWidth = 280
windowHeight = 240

running = True
mtype = 0


def Init():
	global win
	global clock
	global font
	
	pygame.init()
	#logo
	#logo = pygame.image.load("logo.png")
	#pygame.display.set_icon(logo)
	pygame.display.set_caption("Minesweeper")
	
	#create window
	win = pygame.display.set_mode((windowWidth, windowHeight))
	win.fill((40, 40, 40))
	
	clock = pygame.time.Clock()
	
	pygame.font.init()
	font = pygame.font.SysFont("Times New Roman", 100)
	
def InitGame():
	global grid
	m = 20 #margin
	w = int(windowWidth - m*2)
	h = int(windowHeight - m*2)
	grid = Grid.Grid(m, m, w, h, 2)

def MainLoop():	
	win.fill((240, 240, 240))
	tick = clock.tick(60)

	Events()
	Update(tick)
	Draw()
	
	pygame.display.update()
	
def Events():
	global running
	global mtype
	
	# event handling
	for event in pygame.event.get():
		# quit
		if event.type == pygame.QUIT:
			running = False
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if (pygame.mouse.get_pressed()[0]):
				mtype = 0
			elif (pygame.mouse.get_pressed()[2]):
				mtype = 1
		elif event.type == pygame.MOUSEBUTTONUP:
			x, y = pygame.mouse.get_pos()
			grid.Click(x, y, mtype)

def MouseClick():
	# left mouse
	if pygame.mouse.get_pressed()[0]:
		x, y = pygame.mouse.get_pos()
		grid.Click(x, y, 0)
	# right mouse
	elif pygame.mouse.get_pressed()[1]:
		x, y = pygame.mouse.get_pos()
		grid.Click(x, y, 1)

def Update(tick):
	grid.Update()
	
def Draw():
	grid.Draw()
	
def main():	
	Init()
	InitGame()
	
	while running:
		MainLoop()
		
if __name__ == "__main__":
	main()