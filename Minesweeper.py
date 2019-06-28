import pygame

import Globals
import Grid
import Menu

import Screenshot

global _screen #window
global _clock
global _running
global _windowWidth
global _windowHeight

global _mtype

global _SKey
global _GKey

global _grid
global _menu

_screen = None
_clock = None
_running = True

_windowWidth = 280
_windowHeight = 280

_mtype = 0

_SKey = False
_GKey = False

_grid = None
_menu = None

# Training stuff
global _screenshotFonts
global _fontidx
global _screenshotRandomColors
global colorcount
global fonts
_screenshotFonts = False
_fontidx = 0
_screenshotRandomColors = False
_colorcount = 0

def Init():
    global _screen
    global _clock
    
    pygame.init()
    #logo
    #logo = pygame.image.load("logo.png")
    #pygame.display.set_icon(logo)
    pygame.display.set_caption("Minesweeper")
    
    #create window
    _screen = pygame.display.set_mode((_windowWidth, _windowHeight))
    _screen.fill((40, 40, 40))
    
    _clock = pygame.time.Clock()
    
    pygame.font.init()
    
    print("initalize globals")
    Globals._fontname = "times new roman"
    Globals._font = pygame.font.SysFont(Globals._fontname, 12)    
    Globals._screenshot = Screenshot.Screenshot(_screen)

def InitGame():
    global _grid
    global _menu 
    
    m = 20 #margin
    w = int(_windowWidth - m*2)
    h = int(_windowHeight - m*3)
    _grid = Grid.Grid(m, m*2, w, h, 2, _screen)
    _menu = Menu.Menu(_windowWidth, _windowHeight, _screen)
    Globals._gameover = False

def MainLoop():	
    if (Globals._gameover == False):
        _screen.fill((240, 240, 240))
        tick = _clock.tick(60)

        Update(tick)
        Draw()
        MouseHover()
        Events()
        
        # Training
        GetAllFontScreenshots()
        GetColorTraining()
        
        pygame.display.update()
    else:
        InitGame()
    
def Events():
    global _running
    global _mtype
    global _GKey
    global _SKey
    
    # training
    global _screenshotFonts
    global _screenshotRandomColors
    
    # event handling
    for event in pygame.event.get():
        # quit
        if event.type == pygame.QUIT:
            _running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if (pygame.mouse.get_pressed()[0]):
                _mtype = 0
            elif (pygame.mouse.get_pressed()[2]):
                _mtype = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            x, y = pygame.mouse.get_pos()
            _menu.Click(x, y, 0)
            _grid.Click(x, y, _mtype)
        elif event.type == pygame.KEYDOWN:
            if (pygame.key.get_pressed()[pygame.K_g]):
                _GKey = True
            elif (pygame.key.get_pressed()[pygame.K_s]):
                _SKey = True
            elif (pygame.key.get_pressed()[pygame.K_p]):
                GetFontList()
                _screenshotFonts = True
            elif (pygame.key.get_pressed()[pygame.K_l]):
                _screenshotRandomColors = True
        elif event.type == pygame.KEYUP:
            if _GKey:
                _GKey = False
                _SKey = False
                ScreenshotGrid()
            if _SKey:
                _SKey = False
                _GKey = False
                Globals._screenshot.CaptureWindow()
                Globals._screenshot.Save("_Window", "./imgs/")


def MouseHover():
    x, y = pygame.mouse.get_pos()
    _menu.MouseHover(x, y)

def Update(tick):
    _menu.Update(tick)
    _grid.Update(tick)
    
def Draw():
    # optional draw? So I can train a reinforcement network
    _menu.Draw()
    _grid.Draw()

def ScreenshotGrid():
    _grid.Screenshot()
  
def Run():
    Init()
    InitGame()
    
    while _running:
        MainLoop()

# functions below are for training a CNN
    
def GetAllFontScreenshots():
    global _fontidx
    # This is for taking pic of all types of fonts of numbers
    if _screenshotFonts:
        if (_fontidx < len(fonts)):
            Globals._fontname = fonts[_fontidx]
            Globals._font = pygame.font.SysFont(Globals._fontname, 12)
            _fontidx = _fontidx + 1
            ScreenshotGrid()
            
def GetColorTraining():
    global _screenshotRandomColors
    if _screenshotRandomColors:
        _grid.ChangeColors()
        ScreenshotGrid() 
    
def GetFontList():
    # for training data
    global fonts
    from os import listdir
    from os.path import isfile, join
    path = 'fonts/'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    fonts = []
    for fl in files:
        # print(fl)
        sub = fl[1:]
        sub = sub.split('.jpg')
        fonts.append(sub[0])

def main():
    Run()
        
if __name__ == "__main__":
    main()
    