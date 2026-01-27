import tkinter as tk
import numpy as np

class Environment:
    """
    2D grid-based environment representation using NumPy.
    Layout: Ufficio con stanze, corridoi, porte e mobili.
    """

    def __init__(self, width_cells=100, height_cells=80, cell_px=2, inflateValue=2):
        self.width_cells  = int(width_cells)
        self.height_cells = int(height_cells)
        self.cell_px      = int(cell_px)
        self.inflateValue = max(0, int(inflateValue))

        self.border_rects     = []
        self.obstacles_rects  = []

        self._make_default_obstacles()

        self.grid   = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)
        self.gridEx = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)

        self._rasterize_obstacles()

    def _make_default_obstacles(self):
        W, H = self.width_cells, self.height_cells
        
        self.border_rects = [
            (0, 0, W, 1), (0, H-1, W, H), (0, 0, 1, H), (W-1, 0, W, H),
        ]

        #magic geometries
        walls = [
            (0, 34, 15, 36),   
            (20, 34, 45, 36),  
            (55, 34, 75, 36),  
            (90, 34, W, 36),   

            (0, 46, 30, 48),   
            (42, 46, 65, 48),  
            (75, 46, W, 48),   

            (30, 0, 32, 34),   
            (65, 0, 67, 34),   
            (50, 48, 52, H),   
        ]

        furniture = [
            (10, 5, 25, 7), (10, 7, 12, 15), 
            
            (40, 12, 55, 18), 
            (38, 14, 39, 16), (56, 14, 57, 16), # sedie
            
            (72, 5, 74, 15), (80, 20, 82, 30), (88, 5, 90, 15),

            (5, 60, 25, 62), (5, 62, 7, 70), (23, 62, 25, 70),
            
            (60, 60, 85, 65),  (60, 70, 85, 75),
            
            (11, 41, 12, 42), 
            (51, 39, 52, 40), 
            (81, 43, 82, 44), 
            
            (95, 40, 97, 43), (2, 40, 4, 43)
        ]
        
        self.obstacles_rects = walls + furniture

    def _rasterize_obstacles(self):
        W, H = self.width_cells, self.height_cells
        inf = self.inflateValue
        self.grid.fill(0)
        self.gridEx.fill(0)

        for (x0, y0, x1, y1) in self.border_rects:
            self.grid[max(0,y0):min(H,y1), max(0,x0):min(W,x1)] = 1
            self.gridEx[max(0,y0):min(H,y1), max(0,x0):min(W,x1)] = 1

        for (x0, y0, x1, y1) in self.obstacles_rects:
            x0, x1 = max(0, x0), min(W, x1)
            y0, y1 = max(0, y0), min(H, y1)
            
            self.grid[y0:y1, x0:x1] = 1
            ex_y0, ex_y1 = max(0, y0-inf), min(H, y1+inf)
            ex_x0, ex_x1 = max(0, x0-inf), min(W, x1+inf)
            self.gridEx[ex_y0:ex_y1, ex_x0:ex_x1] = 1

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.width_cells and 0 <= cy < self.height_cells

    def is_free(self, cx, cy):
        return self.in_bounds(cx, cy) and self.grid[cy, cx] == 0

    def is_freeEx(self, cx, cy):
        return self.in_bounds(cx, cy) and self.gridEx[cy, cx] == 0

    def neighbors4Ex(self, cx, cy):
        candidates = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        return [(x, y) for (x, y) in candidates if self.is_freeEx(x, y)]

    def neighbors8Ex(self, cx, cy):
        candidates = [
            (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1),
            (cx-1, cy-1), (cx-1, cy+1), (cx+1, cy-1), (cx+1, cy+1)
        ]
        return [(x, y) for (x, y) in candidates if self.is_freeEx(x, y)]

    def set_inflate(self, value):
        self.inflateValue = max(0, int(value))
        self._rasterize_obstacles()

    def draw(self, canvas, show_expanded: bool = False, tag: str = "env"):
        self._draw_grid(canvas, tag)
        self._draw_occupied_cells(canvas, self.grid, "#2f2f2f", tag)
        if show_expanded:
            self._draw_occupied_cells(canvas, self.gridEx, "#9e9e9e", tag)

    def _draw_grid(self, canvas, tag):
        Wpx, Hpx = self.width_cells * self.cell_px, self.height_cells * self.cell_px
        step = 5
        for x in range(0, self.width_cells + 1, step):
            canvas.create_line(x*self.cell_px, 0, x*self.cell_px, Hpx, fill="#e5e5e5", tags=tag)
        for y in range(0, self.height_cells + 1, step):
            canvas.create_line(0, y*self.cell_px, Wpx, y*self.cell_px, fill="#e5e5e5", tags=tag)

    def _draw_occupied_cells(self, canvas, grid, color, tag):
        S = self.cell_px
        occupied_indices = np.argwhere(grid == 1)
        for y, x in occupied_indices:
            canvas.create_rectangle(
                x*S, y*S, (x+1)*S, (y+1)*S,
                fill=color, outline="", tags=tag
            )