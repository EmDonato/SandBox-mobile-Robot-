import tkinter as tk
import math

from environment import Environment
from pathFinding import *
from robot import Robot
from control import *   # rollout + purePursuit
from kalman import Kalman

# --- Global state ---
start = None
goal = None
traj = None
mobileRobot = None
ekf = None             # EKF filter instance

L = 1.0                # lookahead distance
lookahead_idx = 0
running = False
show = False
velRef = 1.0

# PID gains
kp = 0.0
ki = 0.0
kd = 0.0


# ------------------------- GUI callbacks -------------------------
def on_clickPosition(event):
    """Select start and goal with two mouse clicks."""
    global start, goal, mobileRobot, kp, ki, kd, ekf

    cx = event.x // env.cell_px
    cy = event.y // env.cell_px
    cell = (cx, cy)

    if start is None:
        # First click → start
        start = cell
        x0, y0 = start

        mobileRobot = Robot(x0 + 0.5, y0 + 0.5, theta=0.0,
                            kp=kp, ki=ki, kd=kd)

        canvas.create_oval(
            (cx + 0.5) * env.cell_px - 5,
            (cy + 0.5) * env.cell_px - 5,
            (cx + 0.5) * env.cell_px + 5,
            (cy + 0.5) * env.cell_px + 5,
            fill="green", outline=""
        )

        # EKF initialized with starting pose
        ekf = Kalman(L=3.6, dt=0.1, xx=x0+0.5, yy=y0+0.5)
        print("Start:", start)

    elif goal is None:
        # Second click → goal
        goal = cell
        canvas.create_oval(
            (cx + 0.5) * env.cell_px - 5,
            (cy + 0.5) * env.cell_px - 5,
            (cx + 0.5) * env.cell_px + 5,
            (cy + 0.5) * env.cell_px + 5,
            fill="red", outline=""
        )
        print("Goal:", goal)


def on_slide(value):
    global L
    L = float(value)


def on_slideVel(value):
    global velRef
    velRef = float(value)


def on_slidekp(value):
    global kp, mobileRobot
    kp = float(value)
    if mobileRobot:
        mobileRobot.mL.pid.kp = kp
        mobileRobot.mR.pid.kp = kp


def on_slideki(value):
    global ki, mobileRobot
    ki = float(value)
    if mobileRobot:
        mobileRobot.mL.pid.ki = ki
        mobileRobot.mR.pid.ki = ki


def on_slidekd(value):
    global kd, mobileRobot
    kd = float(value)
    if mobileRobot:
        mobileRobot.mL.pid.kd = kd
        mobileRobot.mR.pid.kd = kd


def on_clickshow():
    global show
    show = not show


# ------------------------- Simulation functions -------------------------
def draw_path(canvas, path, cell_px, color="#1976d2"):
    """Draw the planned path on the canvas."""
    if not path or len(path) < 2:
        return
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        px0 = (x0 + 0.5) * cell_px
        py0 = (y0 + 0.5) * cell_px
        px1 = (x1 + 0.5) * cell_px
        py1 = (y1 + 0.5) * cell_px
        canvas.create_line(px0, py0, px1, py1, width=2, fill=color)


def find_lookahead_point(path, est_x, est_y, L):
    """Find the lookahead point using EKF-estimated position."""
    global lookahead_idx
    for i in range(lookahead_idx, len(path)):
        px, py = path[i]
        dx = px - est_x
        dy = py - est_y
        if math.hypot(dx, dy) >= L:
            lookahead_idx = i
            return px, py
    lookahead_idx = len(path) - 1
    return path[-1]


def step():
    """One simulation step."""
    global traj, running, show, velRef, ekf

    if not running or traj is None:
        return
    if mobileRobot.collides_with_env(env):
        print("Collision detected")
        running = False
        return

    v_des = velRef
    dt = 0.1

    # --- EKF state estimate ---
    est_x, est_y, est_theta, est_vL, est_vR = ekf.x_hat.flatten()

    # Find lookahead point
    tx, ty = find_lookahead_point(traj, est_x, est_y, L)

    # Draw lookahead point
    canvas.delete("lookahead")
    S = env.cell_px
    canvas.create_oval(
        (tx + 0.5) * S - 3, (ty + 0.5) * S - 3,
        (tx + 0.5) * S + 3, (ty + 0.5) * S + 3,
        fill="red", outline="", tags="lookahead"
    )

    # --- Controller (uses EKF estimate, not true state) ---
    vdes, wdes = purePursuit(v_des, L, type("dummy", (), {
        "x": est_x, "y": est_y, "theta": est_theta
    })(), [(tx, ty)])

    # Update true robot dynamics
    mobileRobot.step(dt, vdes, wdes)

    # --- EKF update ---
    ekf.predict()
    ekf.update([
        mobileRobot.x_meas,   # noisy GPS-like x
        mobileRobot.y_meas,   # noisy GPS-like y
        mobileRobot.mL.x,     # left wheel state
        mobileRobot.mR.x      # right wheel state
    ])

    # Draw environment and robot
    canvas.delete("env")
    env.draw(canvas, show_expanded=show, tag="env")

    canvas.delete("robot")
    mobileRobot.draw(canvas, env, tag="robot")

    # --- Arrival condition (use estimated state) ---
    dx = est_x - goal[0]
    dy = est_y - goal[1]
    if math.hypot(dx, dy) <= 1.5:
        print("Arrived at goal")
        running = False
        return

    root.after(int(dt * 1000), step)


def start_simulation():
    """Plan path and start simulation."""
    global traj, running

    if start is None or goal is None:
        print("Please select start and goal first!")
        return

    # Path planning
    path = astar(env, start, goal, diagonals=True)
    print("Path found:", len(path), "nodes")
    draw_path(canvas, path, env.cell_px)

    # Rollout trajectory
    v_des = 1.0
    dt = 0.1
    traj = rollout(path, v_des, dt)
    print("Trajectory resampled:", len(traj), "points")

    # Draw robot
    mobileRobot.draw(canvas, env, tag="robot")

    # Start simulation
    running = True
    step()


# ------------------------- Main GUI -------------------------
def main():
    global env, canvas, root

    env = Environment(width_cells=80, height_cells=56, cell_px=10)
    env.set_inflate(3)

    root = tk.Tk()
    root.title("A* + Pure Pursuit + EKF demo")

    Wpx = env.width_cells * env.cell_px
    Hpx = env.height_cells * env.cell_px
    canvas = tk.Canvas(root, width=Wpx, height=Hpx, bg="#fafafa")
    canvas.pack()

    # Draw environment
    env.draw(canvas, show_expanded=False, tag="env")

    # Bind mouse click
    canvas.bind("<Button-1>", on_clickPosition)

    # Sliders
    tk.Scale(root, from_=0.5, to=5.0, resolution=0.1,
             orient="horizontal", command=on_slideVel, label="Velocity").pack(side="left")
    tk.Scale(root, from_=0.5, to=5.0, resolution=0.1,
             orient="horizontal", command=on_slide, label="Lookahead").pack(side="left")
    tk.Scale(root, from_=0.0, to=1.5, resolution=0.1,
             orient="horizontal", command=on_slidekp, label="Kp").pack(side="left")
    tk.Scale(root, from_=0.0, to=2.0, resolution=0.1,
             orient="horizontal", command=on_slideki, label="Ki").pack(side="left")
    tk.Scale(root, from_=0.0, to=0.1, resolution=0.001,
             orient="horizontal", command=on_slidekd, label="Kd").pack(side="left")

    # Buttons
    tk.Button(root, text="Start", command=start_simulation).pack(side="right")
    tk.Button(root, text="Show Occupy Grid", command=on_clickshow).pack(side="right")

    root.mainloop()


if __name__ == "__main__":
    main()
