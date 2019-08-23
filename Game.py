import pygame

import Globals
import Grid
import Menu

import Screenshot

import Models.tnr.TNR as Model

global _screen #window
global _clock
global _running
global _windowWidth
global _windowHeight

global _mtype

global _grid
global _menu

# AI
global _model

_screen = None
_clock = None
_running = True

_windowWidth = 424
_windowHeight = 444

_mtype = 0

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

global FPS
FPS = 60

def Init():
    global _screen
    global _clock
    global _model
    
    print("Initializing pygame")
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
    
    print("Initializing AI")
    _model = Model.NumberClassifier()
    
    print("Initializing globals")
    SetDefaultFont()
    Globals._screenshot = Screenshot.Screenshot(_screen)

def SetDefaultFont():
    Globals._fontname = "times new roman"
    Globals._font = pygame.font.SysFont(Globals._fontname, 24)    

def InitGame():
    global _grid
    global _menu 
    
    m = 20 #margin
    w = int(_windowWidth - m*2)
    h = int(_windowHeight - m*3)
    _grid = Grid.Grid(m, m*2, w, h, 2, _screen)
    _menu = Menu.Menu(_windowWidth, _windowHeight, _screen)
    Globals._gameover = False
    
    print("Ready to play")

def MainLoop():	
    _screen.fill((240, 240, 240))
    tick = _clock.tick(FPS)

    Update(tick)
    Draw()
    MouseHover()
    Events()
    
    # Training
    GetAllFontScreenshots()
    GetColorTraining()
    
    pygame.display.update()
    
    if Globals._newgame:
        Globals._newgame = False
        InitGame()
    
def Events():
    global _running
    global _mtype
    global _GKey
    global _SKey
    global FPS
    
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
            
            if (Globals._gameover == False):
                # Only click if not game over
                _grid.Click(x, y, _mtype)
        elif event.type == pygame.KEYUP:
            if Globals._MakeTrainingData == False:
                if event.key == pygame.K_g:
                    ScreenshotGrid()
                    _model.Predict()
                elif event.key == pygame.K_s:
                    Globals._screenshot.CaptureWindow()
                    Globals._screenshot.Save("_Window", "./imgs/")
                elif event.key == pygame.K_l:
                    _screenshotRandomColors = True
            else:
                print("In training mode can't screenshot normally.")
                if event.key == pygame.K_p:
                    GetFontList()
                    _screenshotFonts = True
                    # FPS = 1
                
            if event.key == pygame.K_t:
                # Training mode.
                Globals._MakeTrainingData = not Globals._MakeTrainingData
                Globals._newgame = True
                
                if Globals._MakeTrainingData:
                    print("Training Mode enabled")
                else:
                    print("Training Mode disabled")

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
    
    if (Globals._gameover):
        Gameover()

def Gameover():
    # display gameover
    textSurface = Globals._font.render(str("Game Over"), False, (0, 0, 0))
    # _screen.blit(textSurface, (_windowWidth/2 - 20, 25))

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
    global _screenshotFonts
    # This is for taking pic of all types of fonts of numbers
    if _screenshotFonts:
        if (_fontidx < len(fonts)):
            Globals._fontname = fonts[_fontidx]
            # Globals._fontname = "hypmokgakbold"
            Globals._font = pygame.font.SysFont(Globals._fontname, 24)
            
            _fontidx = _fontidx + 1
            ScreenshotGrid()
            print(Globals._fontname)
        else:
            _screenshotFonts = False
            _fontidx = 0
            SetDefaultFont()
            
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
    