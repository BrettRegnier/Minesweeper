import pygame

global _font
_font = None
global _fontname
_fontname = ""
global _screenshot
_screenshot = None
global _gameover
_gameover = 0
global _newgame
_newgame = False
global _colorsEnabled
_colorsEnabled = True

# For making training data
global _MakeTrainingData
_MakeTrainingData = False

global _OverrideMineCount
_OverrideMineCount = 0

global _TestCount
_TestCount = 0

def IsGameOver():
    global _gameover
    if _gameover == 0:
        return False
    return True