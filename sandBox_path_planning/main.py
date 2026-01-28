import tkinter as tk
import math
import numpy as np

# ------------------------------------------------------------
# External modules
# ------------------------------------------------------------
# Environment: grid-based world representation with obstacles
# astar / rollout: global path planning and trajectory generation
# Robot: differential-drive robot model with PID motor control
# purePursuit: geometric path-following controller
# Kalman: EKF-based state estimator (x, y, theta)
from environment import Environment
from pathFinding import astar, rollout
from robot import Robot
from control import purePursuit
from kalman import Kalman

# ------------------------------------------------------------
# Global simulation state
# ------------------------------------------------------------
# The simulator is intentionally stateful to keep the GUI logic
# simple and reactive (Tkinter callback driven).
current_active_mode = "NONE"   # NONE, A*, MANUAL

start = None                  # Start cell (grid coordinates)
goal = None                   # Goal cell (grid coordinates)
traj = None                   # Continuous trajectory generated from A*
path = None                   # Discrete A* path (grid cells)

mobileRobot = None            # Robot object (true dynamics)
ekf = None                    # Extended Kalman Filter (state estimate)

running = False               # Simulation loop flag
show_grid = False             # Toggle grid visualization

# Control parameters
velRef = 1.0                  # Reference linear velocity (UI slider)
L = 15.0                      # Lookahead distance for Pure Pursuit
lookahead_idx = 0             # Index optimization for trajectory scan

# PID gains (shared by left/right wheel controllers)
kp, ki, kd = 1.0, 0.2, 0.01

# Keyboard state for manual control (non-blocking)
key_states = {'w': False, 'a': False, 's': False, 'd': False}

# ============================================================
# GUI CALLBACKS
# ============================================================

def reset_simulation():
    """
    Fully reset the simulator state.

    Rationale:
    - Clear all references to planning, estimation and robot state
    - Redraw the environment from scratch
    - Keep UI responsive without restarting the application
    """
    global start, goal, traj, mobileRobot, ekf, lookahead_idx, running, path, current_active_mode

    running = False
    current_active_mode = "NONE"

    start = goal = traj = mobileRobot = ekf = path = None
    lookahead_idx = 0

    canvas.delete("all")
    env.draw(canvas, show_expanded=False, tag="env")

    print("Reset complete.")

def on_clickPosition(event):
    """
    Handle mouse clicks on the canvas.

    First click  -> set start position and initialize robot + EKF
    Second click -> set goal position

    Grid coordinates are derived from pixel coordinates to remain
    consistent with A* and environment discretization.
    """
    global start, goal, mobileRobot, ekf

    cx_int = int(event.x // env.cell_px)
    cy_int = int(event.y // env.cell_px)

    if start is None:
        # Initialize robot at the selected start cell
        start = (cx_int, cy_int)

        mobileRobot = Robot(
            float(cx_int),
            float(cy_int),
            theta=0.0,
            kp=kp,
            ki=ki,
            kd=kd
        )

        # EKF initialized with the robot reference
        # dt matches the GUI update rate (~30 Hz)
        ekf = Kalman(dt=0.033, N=3, robot=mobileRobot)

        S = env.cell_px
        canvas.create_oval(
            cx_int*S-5, cy_int*S-5,
            cx_int*S+5, cy_int*S+5,
            fill="green", tags="start"
        )

    elif goal is None:
        goal = (cx_int, cy_int)
        S = env.cell_px
        canvas.create_oval(
            cx_int*S-5, cy_int*S-5,
            cx_int*S+5, cy_int*S+5,
            fill="red", tags="goal"
        )

def on_key_press(event):
    """Register key press for manual control."""
    key = event.keysym.lower()
    if key in key_states:
        key_states[key] = True

def on_key_release(event):
    """Register key release for manual control."""
    key = event.keysym.lower()
    if key in key_states:
        key_states[key] = False

# ------------------------------------------------------------
# Slider callbacks (runtime tuning)
# ------------------------------------------------------------
def on_slideVel(v):
    global velRef
    velRef = float(v)

def on_slideL(v):
    global L
    L = float(v)

def on_slidekp(v):
    """
    Update proportional gain in real time.

    Both motors share identical PID parameters to preserve
    symmetry in the differential-drive model.
    """
    global kp
    kp = float(v)
    if mobileRobot:
        mobileRobot.mL.pid.kp = kp
        mobileRobot.mR.pid.kp = kp

def on_slideki(v):
    global ki
    ki = float(v)
    if mobileRobot:
        mobileRobot.mL.pid.ki = ki
        mobileRobot.mR.pid.ki = ki

def on_slidekd(v):
    global kd
    kd = float(v)
    if mobileRobot:
        mobileRobot.mL.pid.kd = kd
        mobileRobot.mR.pid.kd = kd

# ============================================================
# CONTROL & SIMULATION LOGIC
# ============================================================

def get_manual_commands():
    """
    Convert keyboard input into velocity commands.

    Design choice:
    - Use v / omega commands instead of wheel speeds
    - Keeps manual and autonomous modes consistent
    """
    v = 0.0
    w = 0.0

    if key_states['w']:
        v = velRef * 20
    elif key_states['s']:
        v = -velRef * 20

    if key_states['a']:
        w = -2.0
    elif key_states['d']:
        w = 2.0

    return v, w

def step():
    """
    Main simulation loop (called at ~30 Hz via Tk.after).

    Pipeline:
    1. Read estimated state from EKF
    2. Compute control commands (manual or Pure Pursuit)
    3. Propagate robot dynamics
    4. Run EKF prediction/update
    5. Render everything
    """
    global traj, running, ekf, mobileRobot, lookahead_idx, current_active_mode

    if not running or ekf is None or mobileRobot is None:
        return

    dt = 0.033

    # Estimated state (used for control, not ground truth)
    est_x = ekf.X[0, 0]
    est_y = ekf.X[1, 0]
    est_theta = ekf.X[2, 0]

    if current_active_mode == "MANUAL":
        vdes, wdes = get_manual_commands()
        tx, ty = est_x, est_y

    else:
        if traj is None:
            return

        # Lookahead target selection
        tx, ty = find_lookahead_point(traj, est_x, est_y, L)

        # Lightweight state object for Pure Pursuit
        est_state = type(
            "State", (), {"x": est_x, "y": est_y, "theta": est_theta}
        )()

        # Pure Pursuit computes v and omega from geometry
        vdes, wdes = purePursuit(
            velRef * 10,
            L,
            est_state,
            [(tx, ty)]
        )

    # Propagate true robot dynamics
    mobileRobot.step(dt, vdes, wdes)

    # EKF prediction + correction
    ekf.run(vdes, wdes)

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------
    S = env.cell_px
    canvas.delete("lookahead", "robot", "kalman_odom")

    if current_active_mode == "A*":
        canvas.create_oval(
            tx*S-3, ty*S-3,
            tx*S+3, ty*S+3,
            fill="orange", tags="lookahead"
        )

    mobileRobot.draw(canvas, env, tag="robot")
    ekf.draw(canvas, S, tag="kalman_odom")

    # Collision / goal checks
    if mobileRobot.collides_with_env(env):
        running = False

    if current_active_mode == "A*" and goal:
        if math.hypot(est_x - goal[0], est_y - goal[1]) <= 10.0:
            running = False

    if running:
        root.after(33, step)

def start_A_star():
    """
    Run A* planning and switch to autonomous mode.

    Rationale:
    - Planning uses the EKF estimate, not the true pose
    - Trajectory rollout converts discrete cells into
      continuous reference points for Pure Pursuit
    """
    global traj, running, lookahead_idx, path, start, goal, current_active_mode

    if start is None or goal is None:
        return

    current_active_mode = "A*"

    s_pos = (
        int(round(ekf.X[0, 0])),
        int(round(ekf.X[1, 0]))
    )

    path = astar(env, s_pos, goal, diagonals=True)

    if path:
        canvas.delete("path")
        draw_path(canvas, path, env.cell_px)

        traj = rollout(path, velRef, 0.05)
        lookahead_idx = 0

        if not running:
            running = True
            step()

def start_manual():
    """
    Switch to manual (teleoperation) mode.

    Previous paths and goals are cleared to avoid ambiguity.
    """
    global running, current_active_mode, start

    if start is None:
        return

    current_active_mode = "MANUAL"
    canvas.delete("path", "goal")

    if not running:
        running = True
        step()

def on_clickshow():
    """Toggle grid visualization."""
    global show_grid
    show_grid = not show_grid

    canvas.delete("env")
    env.draw(canvas, show_expanded=show_grid, tag="env")
    canvas.tag_lower("env")

def draw_path(canvas, path_coords, cell_px):
    """Draw the discrete A* path."""
    for i in range(len(path_coords) - 1):
        x0, y0 = path_coords[i]
        x1, y1 = path_coords[i + 1]
        canvas.create_line(
            x0*cell_px, y0*cell_px,
            x1*cell_px, y1*cell_px,
            width=2, fill="#1976d2", tags="path"
        )

def find_lookahead_point(current_traj, est_x, est_y, look_dist):
    """
    Return the first trajectory point at least look_dist away.

    Optimization:
    - Uses a persistent index to avoid rescanning from zero
    - Assumes monotonic progression along the path
    """
    global lookahead_idx

    for i in range(lookahead_idx, len(current_traj)):
        px, py = current_traj[i]
        if math.hypot(px - est_x, py - est_y) >= look_dist:
            lookahead_idx = i
            return px, py

    return current_traj[-1]

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():
    """
    Initialize environment, GUI and event bindings.
    """
    global env, canvas, root

    env = Environment(
        width_cells=1000,
        height_cells=800,
        cell_px=1,
        scale=10
    )
    env.set_inflate(20)

    root = tk.Tk()
    root.title("SANDBOX MOBILE ROBOT")

    canvas = tk.Canvas(root, width=1000, height=800, bg="#ffffff")
    canvas.pack()

    env.draw(canvas, tag="env")

    # Input bindings
    canvas.bind("<Button-1>", on_clickPosition)
    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)

    # UI controls
    ui = tk.Frame(root)
    ui.pack(fill="x", padx=10, pady=5)

    tk.Scale(ui, from_=0.5, to=10.0, resolution=0.1,
             orient="horizontal", command=on_slideVel,
             label="Vel").pack(side="left")

    tk.Scale(ui, from_=5.0, to=50.0, resolution=1.0,
             orient="horizontal", command=on_slideL,
             label="L").pack(side="left")

    tk.Scale(ui, from_=0.0, to=5.0, resolution=0.05,
             orient="horizontal", command=on_slidekp,
             label="Kp").pack(side="left")

    tk.Scale(ui, from_=0.0, to=2.0, resolution=0.05,
             orient="horizontal", command=on_slideki,
             label="Ki").pack(side="left")

    tk.Scale(ui, from_=0.0, to=2.0, resolution=0.05,
             orient="horizontal", command=on_slidekd,
             label="Kd").pack(side="left")

    # Buttons
    tk.Button(ui, text="START A*", bg="#1976d2", fg="white",
              command=start_A_star, width=12).pack(side="right", padx=5)

    tk.Button(ui, text="START MANUAL", bg="#f57c00", fg="white",
              command=start_manual, width=12).pack(side="right", padx=5)

    tk.Button(ui, text="RESET", bg="#d32f2f", fg="white",
              command=reset_simulation, width=10).pack(side="right", padx=5)

    tk.Button(ui, text="GRID", command=on_clickshow)\
        .pack(side="right", padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
