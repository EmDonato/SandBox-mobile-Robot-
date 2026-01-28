import tkinter as tk
import numpy as np


class Environment:
    """
    Discrete 2D grid-based environment.

    The environment is represented as an occupancy grid where:
    - each cell corresponds to a square region of space,
    - obstacles are rasterized into grid cells,
    - an inflated grid is maintained to account for robot size
      and safety margins (used for planning).
    """

    def __init__(self,
                 width_cells=100,
                 height_cells=80,
                 cell_px=10,
                 inflateValue=2,
                 scale=1):
        """
        Initialize the environment.

        Args:
            width_cells (int): map width in grid cells
            height_cells (int): map height in grid cells
            cell_px (int): size of one grid cell in pixels (rendering only)
            inflateValue (int): obstacle inflation radius in cells
            scale (int): geometric scaling factor for obstacle layout
        """

        # Grid dimensions (discrete planning space)
        self.width_cells  = int(width_cells)
        self.height_cells = int(height_cells)

        # Rendering scale (does NOT affect planning or collision logic)
        self.cell_px = int(cell_px)

        # Obstacle inflation (robot radius + safety margin)
        self.inflateValue = max(0, int(inflateValue))

        # Scaling factor for obstacle geometry
        self.scale = scale

        # Lists of obstacle rectangles (continuous grid coordinates)
        self.border_rects    = []   # outer walls
        self.obstacles_rects = []   # internal obstacles

        # 1) Generate geometric obstacle definitions
        self._make_default_obstacles()

        # 2) Occupancy grid:
        #    grid   -> true obstacles
        #    gridEx -> inflated obstacles (used for planning)
        self.grid   = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)
        self.gridEx = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)

        # 3) Rasterize obstacles into the grids
        self._rasterize_obstacles()

    # ------------------------------------------------------------------
    # Obstacle geometry definition
    # ------------------------------------------------------------------

    def _make_default_obstacles(self):
        """
        Define obstacle geometry using axis-aligned rectangles.

        Obstacles are defined in grid coordinates, not pixels.
        This keeps geometry independent from rendering resolution.
        """

        W, H = self.width_cells, self.height_cells
        s = self.scale

        # Perimeter walls:
        # These guarantee that the robot cannot leave the map.
        self.border_rects = [
            (0,     0,     W,     9),   # top wall
            (0,     H-9,   W,     H),   # bottom wall
            (0,     0,     9,     H),   # left wall
            (W-9,   0,     W,     H),   # right wall
        ]

        # Internal walls (scaled layout)
        # These represent structural obstacles (corridors, partitions)
        walls = [
            (0*s,  34*s, 15*s, 36*s), (20*s, 34*s, 45*s, 36*s),
            (55*s, 34*s, 75*s, 36*s), (90*s, 34*s, W,    36*s),
            (0*s,  46*s, 30*s, 48*s), (42*s, 46*s, 65*s, 48*s),
            (75*s, 46*s, W,    48*s),
            (30*s, 0*s,  32*s, 34*s), (65*s, 0*s,  67*s, 34*s),
            (50*s, 48*s, 52*s, H),
        ]

        # Furniture / clutter:
        # Smaller obstacles that break symmetry and test planner robustness
        furniture = [
            (10*s, 5*s,  25*s, 7*s),  (10*s, 7*s,  12*s, 15*s),
            (40*s, 12*s, 55*s, 18*s), (38*s, 14*s, 39*s, 16*s),
            (56*s, 14*s, 57*s, 16*s), (72*s, 5*s,  74*s, 15*s),
            (80*s, 20*s, 82*s, 30*s), (88*s, 5*s,  90*s, 15*s),
            (5*s,  60*s, 25*s, 62*s), (5*s,  62*s, 7*s,  70*s),
            (23*s, 62*s, 25*s, 70*s), (60*s, 60*s, 85*s, 65*s),
            (60*s, 70*s, 85*s, 75*s), (11*s, 41*s, 12*s, 42*s),
            (51*s, 39*s, 52*s, 40*s), (81*s, 43*s, 82*s, 44*s),
            (95*s, 40*s, 97*s, 43*s), (2*s,  40*s, 4*s,  43*s)
        ]

        # Combine all obstacles except borders
        self.obstacles_rects = walls + furniture

    # ------------------------------------------------------------------
    # Rasterization into occupancy grids
    # ------------------------------------------------------------------

    def _rasterize_obstacles(self):
        """
        Convert obstacle rectangles into occupancy grids.

        grid   : exact obstacle occupancy
        gridEx : inflated obstacle occupancy (used for planning)
        """

        W, H = self.width_cells, self.height_cells
        inf = self.inflateValue

        # Reset grids
        self.grid.fill(0)
        self.gridEx.fill(0)

        # Rasterize all obstacles (borders + internals)
        for (x0, y0, x1, y1) in (self.border_rects + self.obstacles_rects):

            # Clip rectangle bounds to grid limits
            ix0 = int(max(0, x0))
            ix1 = int(min(W, x1))
            iy0 = int(max(0, y0))
            iy1 = int(min(H, y1))

            if ix0 < ix1 and iy0 < iy1:
                # Mark true obstacle cells
                self.grid[iy0:iy1, ix0:ix1] = 1

                # Inflate obstacle for safe planning
                ex_y0 = max(0, iy0 - inf)
                ex_y1 = min(H, iy1 + inf)
                ex_x0 = max(0, ix0 - inf)
                ex_x1 = min(W, ix1 + inf)

                self.gridEx[ex_y0:ex_y1, ex_x0:ex_x1] = 1

    # ------------------------------------------------------------------
    # A* / grid-search helper methods
    # ------------------------------------------------------------------

    def in_bounds(self, cx, cy):
        """Check if a grid coordinate is inside the map."""
        return 0 <= cx < self.width_cells and 0 <= cy < self.height_cells

    def is_freeEx(self, cx, cy):
        """
        Check if a cell is free in the inflated grid.
        This is the condition used by the planner.
        """
        if not self.in_bounds(cx, cy):
            return False
        return self.gridEx[int(cy), int(cx)] == 0

    def neighbors4Ex(self, cx, cy):
        """
        4-connected neighborhood (Manhattan moves).
        Suitable for grid-constrained motion models.
        """
        candidates = [
            (cx-1, cy), (cx+1, cy),
            (cx, cy-1), (cx, cy+1)
        ]
        return [c for c in candidates if self.is_freeEx(c[0], c[1])]

    def neighbors8Ex(self, cx, cy):
        """
        8-connected neighborhood (allows diagonals).
        Produces shorter and smoother paths.
        """
        candidates = [
            (cx-1, cy),   (cx+1, cy),
            (cx, cy-1),   (cx, cy+1),
            (cx-1, cy-1), (cx-1, cy+1),
            (cx+1, cy-1), (cx+1, cy+1)
        ]
        return [c for c in candidates if self.is_freeEx(c[0], c[1])]

    def set_inflate(self, value):
        """
        Update obstacle inflation dynamically and re-rasterize.
        """
        self.inflateValue = max(0, int(value))
        self._rasterize_obstacles()

    # ------------------------------------------------------------------
    # Rendering (visualization only)
    # ------------------------------------------------------------------

    def draw(self, canvas, show_expanded=False, tag="env"):
        """
        Draw environment on a Tkinter canvas.

        Args:
            canvas: Tkinter canvas
            show_expanded (bool): show inflated obstacles (debug)
            tag (str): canvas tag for easy deletion
        """

        S = self.cell_px

        # Draw background grid
        self._draw_grid(canvas, tag)

        # Draw true obstacles
        for (x0, y0, x1, y1) in (self.border_rects + self.obstacles_rects):
            canvas.create_rectangle(
                x0*S, y0*S, x1*S, y1*S,
                fill="#2f2f2f", outline="", tags=tag
            )

        if show_expanded:
            # Visualize inflated obstacles as dashed outlines
            inf = self.inflateValue
            for (x0, y0, x1, y1) in (self.border_rects + self.obstacles_rects):
                canvas.create_rectangle(
                    (x0-inf)*S, (y0-inf)*S,
                    (x1+inf)*S, (y1+inf)*S,
                    outline="#ff0000", dash=(2, 2), tags=tag
                )

    def _draw_grid(self, canvas, tag):
        """
        Draw a background grid to improve spatial readability.
        """
        Wpx = self.width_cells * self.cell_px
        Hpx = self.height_cells * self.cell_px

        # Grid spacing scales with environment resolution
        step = int(5 * self.scale)

        for x in range(0, self.width_cells + 1, step):
            canvas.create_line(
                x*self.cell_px, 0,
                x*self.cell_px, Hpx,
                fill="#e5e5e5", tags=tag
            )

        for y in range(0, self.height_cells + 1, step):
            canvas.create_line(
                0, y*self.cell_px,
                Wpx, y*self.cell_px,
                fill="#e5e5e5", tags=tag
            )
