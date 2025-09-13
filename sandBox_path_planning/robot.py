import math
import numpy as np
from control import PID


class Motor:
    """
    Simple motor model with internal PID controller.
    The motor dynamics are modeled as a first-order system:
        x[k+1] = A * x[k] + B * u[k]
        y[k]   = C * x[k] + noise
    where x is the internal motor state (velocity).
    """

    def __init__(self, kp=1.0, ki=0.5, kd=0.0):
        self.A = 0.2
        self.B = 1.0
        self.C = 1.0
        self.x = 0.0  # internal state (velocity)
        # Dedicated PID controller
        self.pid = PID(kp, ki, kd)

    def step(self, ref, dt):
        """
        Update motor dynamics using PID control.
        Args:
            ref (float): desired velocity reference
            dt  (float): time step
        Returns:
            float: measured motor velocity (with noise)
        """
        error = ref - self.x
        u = self.pid.compute(error, dt)  # PID control action
        self.x = self.A * self.x + self.B * u
        # Return measured velocity with Gaussian noise
        return self.C * self.x + np.random.normal(0, 0.001)


class Robot:
    """
    Differential-drive robot (2D).
    Units are expressed in grid cells for simplicity.
    """

    def __init__(self, x, y, theta=0.0, radius=1.6, wheel_base=3.6,
                 kp=1.0, ki=0.2, kd=0.0):
        # State (pose)
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

        # Geometry
        self.radius = float(radius)          # robot radius [cells]
        self.wheel_base = float(wheel_base)  # wheel separation [cells]

        # Commanded velocities
        self.v_cmd = 0.0  # linear velocity
        self.w_cmd = 0.0  # angular velocity

        # Independent motors (each with its own PID)
        self.mL = Motor(kp, ki, kd)
        self.mR = Motor(kp, ki, kd)

        # Noisy measurements (simulate GPS-like measurements)
        self.x_meas = 0.0
        self.y_meas = 0.0

    def step(self, dt, vdes, wdes):
        """
        Update robot kinematics using differential-drive model.
        Args:
            dt   (float): time step
            vdes (float): desired linear velocity
            wdes (float): desired angular velocity
        """
        # Desired wheel velocities
        Vldes = vdes - wdes * (self.wheel_base / 2.0)
        VRdes = vdes + wdes * (self.wheel_base / 2.0)

        # Motor evolution with internal PID control
        vL = self.mL.step(Vldes, dt)
        vR = self.mR.step(VRdes, dt)

        # Effective robot velocities
        self.v_cmd = (vL + vR) / 2.0
        self.w_cmd = (vR - vL) / self.wheel_base

        # Update robot pose (with process noise)
        self.x += self.v_cmd * math.cos(self.theta) * dt + np.random.normal(0, 0.05)
        self.y += self.v_cmd * math.sin(self.theta) * dt + np.random.normal(0, 0.05)
        self.theta += self.w_cmd * dt + np.random.normal(0, 0.05)

        # GPS-like noisy measurements
        self.x_meas = self.x + np.random.normal(0, 0.025)
        self.y_meas = self.y + np.random.normal(0, 0.025)

        # Normalize theta in [-pi, pi]
        if self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        elif self.theta < -math.pi:
            self.theta += 2.0 * math.pi

    def collides_with_env(self, env):
        """
        Check if the robot collides with the environment.
        Returns:
            bool: True if robot intersects at least one obstacle cell
        """
        r = self.radius
        x_min = int(math.floor(self.x - r))
        x_max = int(math.floor(self.x + r))
        y_min = int(math.floor(self.y - r))
        y_max = int(math.floor(self.y + r))

        W = env.width_cells
        H = env.height_cells

        for cy in range(max(0, y_min), min(H-1, y_max)+1):
            for cx in range(max(0, x_min), min(W-1, x_max)+1):
                if env.grid[cy][cx] == 1:
                    if _circle_rect_intersect(self.x, self.y, r, cx, cy, cx+1.0, cy+1.0):
                        return True
        return False

    def draw(self, canvas, env, tag="robot"):
        """
        Draw the robot (circle + heading + wheels) on a Tkinter canvas.
        """
        S = env.cell_px  # scale [cells → pixels]
        cx = self.x * S
        cy = self.y * S
        rpx = self.radius * S

        # Body (circle)
        canvas.create_oval(
            cx-rpx, cy-rpx, cx+rpx, cy+rpx,
            outline="#212121", width=2, tags=tag
        )

        # Heading line
        hx = cx + rpx * math.cos(self.theta)
        hy = cy + rpx * math.sin(self.theta)
        canvas.create_line(cx, cy, hx, hy, width=3, tags=tag)


def _circle_rect_intersect(cx, cy, r, rx0, ry0, rx1, ry1):
    """Check if circle (cx,cy,r) intersects rectangle [rx0,rx1]x[ry0,ry1]."""
    closest_x = min(max(cx, rx0), rx1)
    closest_y = min(max(cy, ry0), ry1)
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) <= (r*r)
