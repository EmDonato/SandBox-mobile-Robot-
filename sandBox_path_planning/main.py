import tkinter as tk
import math
import numpy as np

# --- Import modules ---
from environment import Environment
from pathFinding import astar, rollout
from robot import Robot
from control import purePursuit
from kalman import Kalman

# --- Global state ---
start = None
goal = None
traj = None
mobileRobot = None
ekf = None
path = None
running = False
show_grid = False
velRef = 1.0
L = 3.0                
lookahead_idx = 0

# PID Initial Gains
kp, ki, kd = 1.0, 0.2, 0.01

# ------------------------- GUI Callbacks -------------------------

def reset_simulation():
    global start, goal, traj, mobileRobot, ekf, lookahead_idx, running, path
    running = False
    start = goal = traj = mobileRobot = ekf = path = None
    lookahead_idx = 0
    canvas.delete("all")
    env.draw(canvas, show_expanded=False, tag="env")
    print("Reset complete.")

def on_clickPosition(event):
    global start, goal, mobileRobot, ekf, traj, lookahead_idx, running, path
    cx, cy = event.x // env.cell_px, event.y // env.cell_px
    
    if start is None:
        start = (cx, cy)
        mobileRobot = Robot(cx, cy, theta=0.0, kp=kp, ki=ki, kd=kd)
        ekf = Kalman(dt=0.033, N=3, robot=mobileRobot)
        canvas.create_oval(cx*10-5, cy*10-5, cx*10+5, cy*10+5, fill="green", tags="start")
        print(f"Start: {start}")
    else:
        goal = (cx, cy)
        canvas.delete("goal")
        canvas.create_oval(cx*10-5, cy*10-5, cx*10+5, cy*10+5, fill="red", tags="goal")
        
        if running:
            # Dynamic re-planning during execution
            curr_est_pos = (int(ekf.X[0,0]), int(ekf.X[1,0]))
            new_path = astar(env, curr_est_pos, goal, diagonals=True)
            if new_path:
                path = new_path
                canvas.delete("path")
                canvas.delete("start")
                draw_path(canvas, path, env.cell_px)
                traj = rollout(path, velRef, 0.1)
                lookahead_idx = 0
                print("Path updated while running.")

def on_clickshow():
    global show_grid
    show_grid = not show_grid
    canvas.delete("env")
    env.draw(canvas, show_expanded=show_grid, tag="env")
    canvas.tag_lower("env")

# --- Sliders Callbacks ---
def on_slideVel(v): global velRef; velRef = float(v)
def on_slideL(v):   global L; L = float(v)
def on_slidekp(v):  
    global kp, mobileRobot; kp = float(v)
    if mobileRobot: mobileRobot.mL.pid.kp = mobileRobot.mR.pid.kp = kp
def on_slideki(v):  
    global ki, mobileRobot; ki = float(v)
    if mobileRobot: mobileRobot.mL.pid.ki = mobileRobot.mR.pid.ki = ki
def on_slidekd(v):  
    global kd, mobileRobot; kd = float(v)
    if mobileRobot: mobileRobot.mL.pid.kd = mobileRobot.mR.pid.kd = kd

# ------------------------- Logic -------------------------

def draw_path(canvas, path_coords, cell_px):
    if not path_coords: return
    for i in range(len(path_coords)-1):
        x0, y0 = path_coords[i]; x1, y1 = path_coords[i+1]
        canvas.create_line(x0*cell_px, y0*cell_px, x1*cell_px, y1*cell_px, 
                           width=2, fill="#1976d2", tags="path")

def find_lookahead_point(current_traj, est_x, est_y, look_dist):
    global lookahead_idx
    for i in range(lookahead_idx, len(current_traj)):
        px, py = current_traj[i]
        if math.hypot(px - est_x, py - est_y) >= look_dist:
            lookahead_idx = i
            return px, py
    return current_traj[-1]

def step():
    global traj, running, ekf, mobileRobot, lookahead_idx
    if not running or traj is None or ekf is None: return
    
    dt = 0.033
    est_x, est_y, est_theta = ekf.X[0,0], ekf.X[1,0], ekf.X[2,0]
    tx, ty = find_lookahead_point(traj, est_x, est_y, L)
    
    est_state = type("State", (), {"x": est_x, "y": est_y, "theta": est_theta})()
    vdes, wdes = purePursuit(velRef, L, est_state, [(tx, ty)])

    mobileRobot.step(dt, vdes, wdes)
    ekf.run()

    S = env.cell_px
    canvas.delete("lookahead", "robot", "kalman_odom")
    canvas.create_oval(tx*S-3, ty*S-3, tx*S+3, ty*S+3, fill="orange", tags="lookahead")
    mobileRobot.draw(canvas, env, tag="robot")
    ekf.draw(canvas, S, tag="kalman_odom")

    if mobileRobot.collides_with_env(env):
        print("Collision!"); running = False
    
    if math.hypot(est_x - goal[0], est_y - goal[1]) <= 1.5:
        print("Goal reached!"); running = False

    if running: root.after(33, step)

def start_simulation():
    global traj, running, lookahead_idx, path, start
    if start is None or goal is None: return
    s_pos = (int(ekf.X[0,0]), int(ekf.X[1,0])) if ekf else start
    path = astar(env, s_pos, goal, diagonals=True)
    if path:
        canvas.delete("path")
        draw_path(canvas, path, env.cell_px)
        traj = rollout(path, velRef, 0.1)
        lookahead_idx = 0
        if not running:
            running = True; step()

# ------------------------- Main -------------------------

def main():
    global env, canvas, root
    env = Environment(width_cells=100, height_cells=80, cell_px=10)
    env.set_inflate(2.7)#add padding
    
    root = tk.Tk()
    root.title("SANDBOX MOBILE ROBOT")

    canvas = tk.Canvas(root, width=1000, height=800, bg="#ffffff")
    canvas.pack()
    env.draw(canvas, tag="env")
    canvas.bind("<Button-1>", on_clickPosition)

    # UI Panel
    ui = tk.Frame(root)
    ui.pack(fill="x", padx=10, pady=5)
    
    # Sliders
    tk.Scale(ui, from_=0.5, to=5.0, resolution=0.1, orient="horizontal", command=on_slideVel, label="Vel").pack(side="left")
    tk.Scale(ui, from_=0.5, to=8.0, resolution=0.1, orient="horizontal", command=on_slideL, label="L").pack(side="left")
    tk.Scale(ui, from_=0.0, to=2.0, resolution=0.05, orient="horizontal", command=on_slidekp, label="Kp").pack(side="left")
    tk.Scale(ui, from_=0.0, to=2.0, resolution=0.05, orient="horizontal", command=on_slideki, label="Ki").pack(side="left")
    tk.Scale(ui, from_=0.0, to=0.5, resolution=0.01, orient="horizontal", command=on_slidekd, label="Kd").pack(side="left")
    
    # Buttons
    tk.Button(ui, text="START", bg="#2e7d32", fg="white", command=start_simulation, width=10).pack(side="right", padx=5)
    tk.Button(ui, text="RESET", bg="#d32f2f", fg="white", command=reset_simulation, width=10).pack(side="right", padx=5)
    tk.Button(ui, text="GRID", command=on_clickshow).pack(side="right", padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()