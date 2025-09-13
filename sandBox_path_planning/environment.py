# environment.py
import tkinter as tk

class Environment:
    """
    2D grid-based environment representation.

    Features:
    - Occupancy grid (0 = free, 1 = occupied) -> `grid`
    - Inflated occupancy grid (with obstacle dilation) -> `gridEx`
    - Default rectangular obstacles (outer border + internal blocks)
    - API for planners: boundary check, occupancy queries, 4/8-connectivity neighbors
    - Visualization on a Tkinter Canvas (with optional overlay of inflated obstacles)

    Units: expressed in grid cells (not pixels).
    """

    def __init__(self, width_cells=40, height_cells=28, cell_px=20, inflateValue=1):
        self.width_cells  = int(width_cells)
        self.height_cells = int(height_cells)
        self.cell_px      = int(cell_px)
        self.inflateValue = max(0, int(inflateValue))

        # Obstacle lists
        self.border_rects     = []  # outer frame (NOT inflated)
        self.obstacles_rects  = []  # internal obstacles (inflated)

        # Initialize default obstacles
        self._make_default_obstacles()

        # Occupancy grids
        self.grid   = self._init_empty_grid()  # base occupancy
        self.gridEx = self._init_empty_grid()  # inflated occupancy

        # Rasterize obstacles into the grids
        self._rasterize_obstacles()

    # ---------- Obstacle setup ----------
    def _make_default_obstacles(self):
        """Define outer borders and some example internal rectangular obstacles."""
        W, H = self.width_cells, self.height_cells

        # Outer borders (not inflated in gridEx)
        self.border_rects = [
            (0, 0, W, 1),         # top edge
            (0, H-1, W, H),       # bottom edge
            (0, 0, 1, H),         # left edge
            (W-1, 0, W, H),       # right edge
        ]

        # Example internal blocks (these will be inflated in gridEx)
        self.obstacles_rects = [
            (12,  8, 36, 12),
            (20, 20, 60, 24),
            (44, 32, 48, 52),
            (8, 36, 28, 40),
            (56, 12, 64, 28),
            (55, 35, 63, 43),
        ]

    def _init_empty_grid(self):
        """Create an empty occupancy grid initialized with free cells (0)."""
        return [[0 for _ in range(self.width_cells)]
                   for _ in range(self.height_cells)]

    def _rasterize_obstacles(self):
        """Fill the grids (`grid` and `gridEx`) with border and obstacle information."""
        W, H = self.width_cells, self.height_cells
        inf = self.inflateValue

        # Reset both grids
        self.grid   = self._init_empty_grid()
        self.gridEx = self._init_empty_grid()

        # 1) Borders: always occupied in both grids, no inflation
        for (x0, y0, x1, y1) in self.border_rects:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    self.grid[y][x] = 1
                    self.gridEx[y][x] = 1  # same cell in inflated grid

        # 2) Internal obstacles: placed in `grid` and inflated in `gridEx`
        for (x0, y0, x1, y1) in self.obstacles_rects:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)

            # Mark original obstacle in base grid
            for y in range(y0, y1):
                for x in range(x0, x1):
                    self.grid[y][x] = 1

            # Mark inflated obstacle in expanded grid
            ex_x0 = x0 - inf
            ex_y0 = y0 - inf
            ex_x1 = x1 + inf
            ex_y1 = y1 + inf
            for y in range(ex_y0, ex_y1):
                if 0 <= y < H:
                    for x in range(ex_x0, ex_x1):
                        if 0 <= x < W:
                            self.gridEx[y][x] = 1

    # ---------- API for path planners ----------
    def in_bounds(self, cx, cy):
        """Check if cell coordinates (cx, cy) are inside map boundaries."""
        return 0 <= cx < self.width_cells and 0 <= cy < self.height_cells

    def is_free(self, cx, cy):
        """Return True if cell (cx, cy) is free in the base grid."""
        return self.in_bounds(cx, cy) and self.grid[cy][cx] == 0

    def is_freeEx(self, cx, cy):
        """Return True if cell (cx, cy) is free in the inflated grid."""
        return self.in_bounds(cx, cy) and self.gridEx[cy][cx] == 0

    def neighbors4(self, cx, cy):
        """Return 4-connected neighbors (up, down, left, right) in base grid."""
        cand = [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]
        return [(x, y) for (x, y) in cand if self.is_free(x, y)]

    def neighbors4Ex(self, cx, cy):
        """Return 4-connected neighbors in inflated grid."""
        cand = [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]
        return [(x, y) for (x, y) in cand if self.is_freeEx(x, y)]

    def neighbors8(self, cx, cy):
        """Return 8-connected neighbors (diagonals included) in base grid."""
        cand = [
            (cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1),
            (cx-1,cy-1), (cx-1,cy+1), (cx+1,cy-1), (cx+1,cy+1)
        ]
        return [(x, y) for (x, y) in cand if self.is_free(x, y)]

    def neighbors8Ex(self, cx, cy):
        """Return 8-connected neighbors in inflated grid."""
        cand = [
            (cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1),
            (cx-1,cy-1), (cx-1,cy+1), (cx+1,cy-1), (cx+1,cy+1)
        ]
        return [(x, y) for (x, y) in cand if self.is_freeEx(x, y)]

    def set_inflate(self, value):
        """Update obstacle inflation factor and recompute inflated grid."""
        self.inflateValue = max(0, int(value))
        self._rasterize_obstacles()

    # ---------- Visualization ----------
    def draw(self, canvas, show_expanded: bool = False, tag: str = "env"):
        """
        Draw the environment on a Tkinter Canvas.
        - Base grid lines
        - Obstacles (dark gray)
        - Optionally overlay inflated obstacles (light gray)
        All objects are drawn with the same `tag` for easy deletion.
        """
        self._draw_grid(canvas, tag)
        self._draw_occupied_cells(canvas, self.grid, "#2f2f2f", tag)  # base obstacles
        if show_expanded:
            self._draw_occupied_cells(canvas, self.gridEx, "#9e9e9e", tag)  # inflated overlay

    def _draw_grid(self, canvas, tag):
        """Draw grid lines on the canvas."""
        Wpx = self.width_cells * self.cell_px
        Hpx = self.height_cells * self.cell_px
        for x in range(self.width_cells):
            x0 = x * self.cell_px
            canvas.create_line(x0, 0, x0, Hpx, fill="#e5e5e5", tags=tag)
        for y in range(self.height_cells):
            y0 = y * self.cell_px
            canvas.create_line(0, y0, Wpx, y0, fill="#e5e5e5", tags=tag)

    def _draw_occupied_cells(self, canvas, grid, color, tag):
        """Draw occupied cells from a given occupancy grid."""
        S = self.cell_px
        for y in range(self.height_cells):
            for x in range(self.width_cells):
                if grid[y][x] == 1:
                    canvas.create_rectangle(
                        x*S, y*S, (x+1)*S, (y+1)*S,
                        fill=color, outline="", tags=tag
                    )
