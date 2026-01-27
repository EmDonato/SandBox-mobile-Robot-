import numpy as np
import math

def normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class Kalman:
    def __init__(self, dt, N, robot, a_coeff=0.0001):
        self.dt = dt
        self.N = N
        self.robot = robot
        self.a = a_coeff
        self.i = 0
        self.j = 0
        self.N_gps = N * 5  

        # State: [x, y, theta, omega, bias] actually we know bias
        self.X = np.array([
            [robot.x],
            [robot.y],
            [robot.theta],
            [0.0],          
            [0.0001]        
        ])

        self.P = np.eye(5) * 1e-2
        self.Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-5]) #Uncetancy model and encoders
        self.R_imu = np.array([[0.025**2]]) #Uncetancy IMU Theta
        self.R_gps = np.diag([0.05**2, 0.05**2]) #Uncetancy GPS  X e Y

    def run(self):
        
        self.update_imu(self.robot.yaw_rate_)#fast update with imu

        #prediction with encoder 100hz
        self.i = (self.i + 1) % self.N
        if self.i == 0:
            dt_enc = self.dt * self.N
            vl, vr = self.robot.get_enc(self.N)
            v_enc = (vl + vr) / 2.0
            w_enc = (vr - vl) / self.robot.wheel_base
            self.predict_encoder(v_enc, w_enc, dt_enc)

        # sloooooow update: GPS 165hz
        self.j = (self.j + 1) % self.N_gps
        if self.j == 0:
            self.update_gps()

    def predict_encoder(self, v, w_enc, dt):
        x, y, theta, omega, bias = self.X.flatten()

        # state Update 
        self.X[0, 0] = x + v * math.cos(theta) * dt
        self.X[1, 0] = y + v * math.sin(theta) * dt
        self.X[2, 0] = normalize_angle(theta + w_enc * dt)
        self.X[3, 0] = w_enc 
        self.X[4, 0] = bias + self.a * dt

        # Jacobian F
        F = np.eye(5)
        F[0, 2] = -v * math.sin(theta) * dt
        F[1, 2] =  v * math.cos(theta) * dt
        F[2, 3] = dt 
        self.P = F @ self.P @ F.T + self.Q

    def update_imu(self, z_gyro):
        z = np.array([[z_gyro]])
        h = self.X[3, 0] + self.X[4, 0] # omega + bias
        H = np.array([[0, 0, 0, 1, 1]])

        y_err = z - h
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ y_err
        self.X[2, 0] = normalize_angle(self.X[2, 0])
        self.P = (np.eye(5) - K @ H) @ self.P

    def update_gps(self):
        # [x_gps, y_gps]
        z = np.array([[self.robot.x_gps], 
                      [self.robot.y_gps]])
        
        # h(x) = [x, y] 
        h = self.X[0:2]
        
        
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])

        y_err = z - h
        
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ y_err
        self.P = (np.eye(5) - K @ H) @ self.P

    def draw(self, canvas, cell_px, tag="kalman_odom"): #our belive
        S = cell_px
        x, y, theta = self.X[0,0], self.X[1,0], self.X[2,0]
        cx, cy = x * S, y * S
        rpx = self.robot.radius * S

        canvas.delete(tag)
        canvas.create_oval(cx-rpx, cy-rpx, cx+rpx, cy+rpx, outline="red", width=2, tags=tag)
        hx = cx + rpx * math.cos(theta)
        hy = cy + rpx * math.sin(theta)
        canvas.create_line(cx, cy, hx, hy, fill="red", width=2, tags=tag)