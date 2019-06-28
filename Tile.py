import pygame
import Globals

import random

class Tile:
    def __init__(self, x, y, size, id, screen, count):
        padding = len(str(count))
        self._id = id
        self._strid = str(id).rjust(padding, '0')
        self._x = x
        self._y = y
        self._size = size
        self._screen = screen
        
        self._mine = False
        
        # tuple of adjacentTiles
        self._nearbyMines = 0
        
        self._revealed = False
        self._flagged = False
        
        self._outercolor = 255
        self._innercolor = 200        
        self._revealedcolor = 180        
        self._outline = 100
        
        # For training
        self._colorTraining = False
        
    # for training purposes
    def OverrideValue(self, val):
        self._nearbyMines = val
        self._revealed = True
    
    def BeMine(self):
        self._mine = True
        
    def IsMine(self):
        return self._mine
        
    def AdjacentTiles(self, adj):
        self._adjacentTiles = adj
        
        for tile in self._adjacentTiles:
            if (tile is not None and tile.IsMine()):
                self._nearbyMines = self._nearbyMines + 1
    
    def Draw(self):
        #border
        pygame.draw.rect(self._screen, 
        (self._outline, self._outline, self._outline), 
        (self._x, self._y, self._size, self._size))
        
        # revealed tile
        if (self._revealed):
            c = self._revealedcolor
            nx = self._x + 1
            ny = self._y + 1
            ns = self._size -2
            
            pygame.draw.rect(self._screen, (c, c, c), (nx, ny, ns, ns))
            
            if self._colorTraining:
                r = random.randint(0,255)
                g = random.randint(0,255)
                b = random.randint(0,255)
                pygame.draw.rect(self._screen, (r, g, b), (nx, ny, ns, ns))
        else:
            # decor
            ci = self._innercolor
            co = self._outercolor
            
            # outer design
            nx = self._x + 1
            ny = self._y + 1
            ns = self._size - 2
            pygame.draw.rect(self._screen, 
            (co, co, co), 
            (nx, ny, ns, ns))
            
            # inner design
            nx = self._x + 3
            ny = self._y + 3
            ns = self._size - 4
            pygame.draw.rect(self._screen, 
            (ci, ci, ci), 
            (nx, ny, ns, ns))
            
            if self._colorTraining:
                r = random.randint(0,255)
                g = random.randint(0,255)
                b = random.randint(0,255)
                pygame.draw.rect(self._screen, (r, g, b), (nx, ny, ns, ns))
        
        if (self._revealed and self._mine):
            # has a mine
            pygame.draw.ellipse(self._screen,
            (0, 0, 0), 
            (nx+2, ny+2, 
            ns-4, ns-4))
        elif (self._revealed and self._nearbyMines > 0):
            # mines nearby
            textSurface = Globals._font.render(str(self._nearbyMines), False, (0, 0, 0))
            self._screen.blit(textSurface, (self._x + 8, self._y + 5))
        
        # flag
        if (self._flagged and not self._revealed):
            cx = int(self._x + (self._size / 2))
            cy = int(self._y + (self._size / 2))
            tip = int(self._size / 2 - 4)
            
            pygame.draw.polygon(self._screen,
            (255, 0, 0),
            [[cx, ny+2], [cx, cy+2], [cx+tip, cy]])
            
            pygame.draw.line(self._screen,
            (255, 0, 0),
            [cx, cy], [cx, self._y + self._size - 3])
        
    def Update(self, tick):
        #animation stuff here.
        pass
        
    def Click(self, mx, my, mtype):
        if (mx > self._x  and mx < self._x + self._size and 
        my > self._y and my < self._y + self._size):
            if (mtype == 0 and self._revealed == False):
                #left click
                self.Reveal()
                
                # Test
                # if (not self._mine):
                # 	print(self._nearbyMines)
                #
            elif (mtype == 1):
                #right click
                self._flagged = not self._flagged
    
    def Reveal(self):
        self._revealed = True
        
        #only cascade if the tile is blank
        if (self._nearbyMines == 0 and not self._mine):
            self.Cascade()
        
    def Revealed(self):
        return self._revealed
        
    def Cascade(self):
        for tile in self._adjacentTiles:
            if (tile is not None and not tile.Revealed()):
                tile.Reveal()
     
    def Screenshot(self):
        if self._id < 9 and not self._id == 0:
            Globals._screenshot.Capture(self._x, self._y, self._size, self._size)
            Globals._screenshot.Save(str(self._strid) + Globals._fontname, "./dataset/gameset/")
            
        # Globals._screenshot.Capture(self._x, self._y, self._size, self._size)
        # Globals._screenshot.Save(str(self._id), "./dataset/gameset/")
    
    def RandomizeColor(self):
        self._colorTraining = True
        