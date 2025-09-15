import math

try:
    from scipy.interpolate import CubicSpline  # Optional: used for smooth trajectory generation
except Exception:
    CubicSpline = None

import heapq


def manhattan(a, b):
    """
    Manhattan distance heuristic.
    Args:
        a (tuple): (x, y) coordinates of node A.
        b (tuple): (x, y) coordinates of node B.
    Returns:
        float: Manhattan distance between A and B.
    """
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)


def octile(a, b):
    """
    Octile distance heuristic (used when diagonal moves are allowed).
    Args:
        a (tuple): (x, y) coordinates of node A.
        b (tuple): (x, y) coordinates of node B.
    Returns:
        float: Octile distance between A and B.
    """
    ax, ay = a
    bx, by = b
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)


def reconstruct_path(came_from, current):
    """
    Reconstructs the path by backtracking from the goal to the start.
    Args:
        came_from (dict): Mapping of nodes to their parent node.
        current (tuple): Current node (goal).
    Returns:
        list: Reconstructed path [(x,y), ...].
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(env, start, goal, diagonals=True):
    """
    A* pathfinding algorithm on a grid-based environment.

    Args:
        env (Environment): The grid environment (provides free/occupied info).
        start (tuple): Starting cell (x, y).
        goal (tuple): Goal cell (x, y).
        diagonals (bool): If True, allows diagonal moves (8-connectivity).

    Returns:
        list or None: Path [(x,y), ...] if found, otherwise None.
    """
    sx, sy = start
    gx, gy = goal

    # Check if start/goal are free cells
    if not env.is_freeEx(sx, sy):
        return None
    if not env.is_freeEx(gx, gy):
        return None

    # Select heuristic and neighbors function
    h = octile if diagonals else manhattan
    neighbors = env.neighbors8Ex if diagonals else env.neighbors4Ex

    # Open set (priority queue)
    open_heap = []
    heapq.heappush(open_heap, (0.0, (sx, sy)))

    g = {(sx, sy): 0.0}   # Cost to reach each node
    came_from = {}        # Parent map
    closed = set()        # Closed set

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        # Goal reached
        if current == (gx, gy):
            return reconstruct_path(came_from, current)

        closed.add(current)
        cx, cy = current

        for (nx, ny) in neighbors(cx, cy):
            if (nx, ny) in closed:
                continue

            # Step cost: diagonal = √2, straight = 1
            step = math.sqrt(2.0) if (nx != cx and ny != cy) else 1.0
            tentative = g[current] + step
            old = g.get((nx, ny), float("inf"))

            # If a better path is found
            if tentative < old:
                g[(nx, ny)] = tentative
                came_from[(nx, ny)] = current
                f = tentative + h((nx, ny), (gx, gy))
                heapq.heappush(open_heap, (f, (nx, ny)))

    return None  # No path found


def make_splines(path):
    """
    Creates cubic splines x(s), y(s) parameterized by arc-length s.

    Args:
        path (list): Sequence of waypoints [(x,y), ...].

    Returns:
        tuple: (s, x_spline, y_spline)
            - s: cumulative arc-length list
            - x_spline: spline function for x(s)
            - y_spline: spline function for y(s)
    """
    x = [p[0] for p in path]
    y = [p[1] for p in path]

    # Compute cumulative arc length
    s = [0.0]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))

    # Build cubic splines
    x_s = CubicSpline(s, x)
    y_s = CubicSpline(s, y)

    return s, x_s, y_s


def rollout(path, v_des, dt):
    """
    Samples a path at constant velocity to create a time-parameterized trajectory.
    
    Args:
        path (list): List of waypoints [(x,y), ...].
        v_des (float): Desired velocity along the path [m/s].
        dt (float): Sampling period [s].
    
    Returns:
        list: Resampled trajectory [(x,y), ...].
    """
    s, x_s, y_s = make_splines(path)
    S_tot = s[-1]            # Total arc length
    T_tot = S_tot / v_des    # Total traversal time

    traj = []
    t = 0.0
    while t <= T_tot + 1e-9:   # Add a small margin to include last point
        s_curr = min(v_des * t, S_tot)
        traj.append((float(x_s(s_curr)), float(y_s(s_curr))))
        t += dt

    return traj

