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
        self.A = 1.0
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
        return self.C * self.x 


class Robot:
    """
    Differential-drive robot (2D).
    Units are expressed in grid cells for simplicity.
    """

    def __init__(self, x, y, theta=0.0, radius=1.2, wheel_base=2.4,
                 kp=1.0, ki=0.2, kd=0.0):
        # State (pose)
        self.x = float(x)
        self.y = float(y)
        self.x_gps = float(x)
        self.y_gps = float(y)
        self.theta = float(theta)
        self.bias = 0.0017
        # Geometry
        self.radius = float(radius)          # robot radius [cells]
        self.wheel_base = float(wheel_base)  # wheel separation [cells]

        # Commanded velocities
        self.v_cmd = 0.0  # linear velocity
        self.w_cmd = 0.0  # angular velocity


        # Independent motors (each with its own PID)
        self.mL = Motor(kp, ki, kd)
        self.mR = Motor(kp, ki, kd)

        #encoders comulative
        self.vL_enc_ = 0.0
        self.vR_enc_ = 0.0
        #imu
        self.bias_ = 0.0017
        self.yaw_rate_ = 0.0
        self.a_ = 0.0001
        self.w_true_ = 0.0  # angular velocity



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
        self.w_true_ = (vR - vL) / self.wheel_base
        vL = vL + np.random.normal(0, 0.05)
        vR = vR + np.random.normal(0, 0.05)
        # Effective robot velocities
        self.v_cmd = (vL + vR) / 2.0
        self.w_cmd = (vR - vL) / self.wheel_base

        # Update robot pose realpose
        self.x += self.v_cmd * math.cos(self.theta) * dt 
        self.y += self.v_cmd * math.sin(self.theta) * dt 
        self.theta += self.w_cmd * dt

        #gps
        self.x_gps = self.x + np.random.normal(0, 0.05)
        self.y_gps = self.y + np.random.normal(0, 0.05)
         
        #enc
        self.upload_enc_(vL, vR)

        #imu
        self.upload_imu_(self.w_true_, dt)


        # GPS-like noisy measurements
        #self.x_meas = self.x + np.random.normal(0, 0.025)
        #self.y_meas = self.y + np.random.normal(0, 0.025)

        # Normalize theta in [-pi, pi]
        if self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        elif self.theta < -math.pi:
            self.theta += 2.0 * math.pi
    def upload_enc_(self, vel_left, vel_right):
        self.vL_enc_ += vel_left
        self.vR_enc_ += vel_right 

    def get_enc(self, N):
            # Calcola il risultato prima di resettare!
            res_L = self.vL_enc_ / N
            res_R = self.vR_enc_ / N
            
            # Ora resetta gli accumulatori per i prossimi N campioni
            self.vL_enc_ = 0.0
            self.vR_enc_ = 0.0
            
            return res_L, res_R
    def upload_bias_(self, dt):
        self.bias_ = self.bias_  + self.a_*dt

    def upload_imu_(self, w, dt):
        self.yaw_rate_ = w + self.bias_+ np.random.normal(0, 0.025)
        self.upload_bias_(dt)


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
