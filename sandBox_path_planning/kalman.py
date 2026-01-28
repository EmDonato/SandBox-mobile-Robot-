import numpy as np
import math


# ==========================================================
# Utility
# ==========================================================

def normalize_angle(a):
    """
    Normalize angle to [-pi, pi].

    Required because angles live on SO(2).
    Without normalization, EKF linearization
    would break due to angle wrapping.
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ==========================================================
# EXTENDED KALMAN FILTER
# ==========================================================

class Kalman:
    """
    Extended Kalman Filter for a differential-drive robot.

    STATE VECTOR:
        X = [x, y, theta, omega, bias]^T

    where:
        x, y   : position
        theta  : heading angle
        omega  : true angular velocity
        bias   : gyroscope bias (slow random walk)

    DESIGN CHOICES:
    - Linear velocity v is NOT part of the state.
      It is treated as a known input (encoder-based).
    - Angular velocity omega IS a state to allow
      fusion between encoder and IMU.
    - Bias is explicitly estimated to prevent gyro drift.
    """

    def __init__(self, dt, N, robot):
        """
        Args:
            dt (float): EKF time step [s]
            N  (int): encoder decimation factor
            robot (object): robot providing sensor data
        """

        self.dt = dt
        self.N = N
        self.robot = robot
        self.i = 0  # counter for encoder updates

        # --------------------------------------------------
        # STATE INITIALIZATION
        # --------------------------------------------------
        self.X = np.array([
            [robot.x],
            [robot.y],
            [robot.theta],
            [0.0],   # omega
            [0.0]    # gyro bias
        ])

        # --------------------------------------------------
        # STATE COVARIANCE
        # --------------------------------------------------
        self.P = np.eye(5) * 1e-2

        # --------------------------------------------------
        # PROCESS NOISE
        # --------------------------------------------------
        # Models uncertainty of the physical evolution
        self.Q = np.diag([
            1e-4,   # x
            1e-4,   # y
            1e-4,   # theta
            1e-3,   # omega
            1e-6    # bias
        ])

        # --------------------------------------------------
        # MEASUREMENT NOISE
        # --------------------------------------------------
        # IMU: z = omega + bias
        self.R_imu = np.array([[0.025**2]])

        # Encoder: z = omega
        self.R_enc = np.array([[0.02**2]])

    # ======================================================
    # MAIN FILTER STEP
    # ======================================================

    def run(self, v_des, w_des):
        """
        Execute one EKF iteration.

        Args:
            v_des (float): commanded linear velocity
            w_des (float): commanded angular velocity
        """

        # 1) Prediction using motion model
        self.predict_model(v_des, w_des, self.dt)

        # 2) IMU update (high-rate)
        self.update_imu(self.robot.yaw_rate_)

        # 3) Encoder update (lower-rate)
        self.i = (self.i + 1) % self.N
        if self.i == 0:
            vl, vr = self.robot.get_enc(self.N)
            w_meas = (vr - vl) / self.robot.wheel_base
            self.update_encoder(w_meas)
        self.display_status()
    # ======================================================
    # PREDICTION STEP
    # ======================================================

    def predict_model(self, v, w, dt):
        """
        Nonlinear prediction based on unicycle kinematics.

        x_{k+1}     = x_k + v cos(theta) dt
        y_{k+1}     = y_k + v sin(theta) dt
        theta_{k+1} = theta_k + omega dt
        omega_{k+1} = w_des
        bias_{k+1}  = bias_k

        ASSUMPTIONS:
        - v is known from encoders
        - omega tracks commanded value
        - bias follows a random walk
        """

        x, y, theta, omega, bias = self.X.flatten()

        # State propagation
        self.X[0, 0] = x + v * math.cos(theta) * dt
        self.X[1, 0] = y + v * math.sin(theta) * dt
        self.X[2, 0] = normalize_angle(theta + omega * dt)
        self.X[3, 0] = w
        self.X[4, 0] = bias

        # Jacobian of the motion model
        F = np.eye(5)
        F[0, 2] = -v * math.sin(theta) * dt
        F[1, 2] =  v * math.cos(theta) * dt
        F[2, 3] = dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    # ======================================================
    # ENCODER UPDATE (omega)
    # ======================================================

    def update_encoder(self, w_meas):
        """
        Encoder measurement update.

        Measurement model:
            z = omega + noise
        """

        z = np.array([[w_meas]])
        h = np.array([[self.X[3, 0]]])

        H = np.array([[0, 0, 0, 1, 0]])

        y = z - h
        S = H @ self.P @ H.T + self.R_enc
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ y
        self.X[2, 0] = normalize_angle(self.X[2, 0])
        self.P = (np.eye(5) - K @ H) @ self.P

    # ======================================================
    # IMU UPDATE (gyro)
    # ======================================================

    def update_imu(self, z_gyro):
        """
        IMU update.

        Measurement model:
            z = omega + bias + noise

        Makes the gyro bias observable.
        """

        z = np.array([[z_gyro]])
        h = np.array([[self.X[3, 0] + self.X[4, 0]]])

        H = np.array([[0, 0, 0, 1, 1]])

        y = z - h
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ y
        self.X[2, 0] = normalize_angle(self.X[2, 0])
        self.P = (np.eye(5) - K @ H) @ self.P

    # ======================================================
    # DEBUG / VISUALIZATION
    # ======================================================

    def draw(self, canvas, cell_px, tag="kalman_odom"):
        """
        Draw EKF-estimated robot pose on a Tkinter canvas.
        """
        S = cell_px
        x, y, theta = self.X[0, 0], self.X[1, 0], self.X[2, 0]
        cx, cy = x * S, y * S
        rpx = self.robot.radius * S

        canvas.delete(tag)

        canvas.create_oval(
            cx - rpx, cy - rpx,
            cx + rpx, cy + rpx,
            outline="red", width=2, tags=tag
        )

        hx = cx + rpx * math.cos(theta)
        hy = cy + rpx * math.sin(theta)

        canvas.create_line(
            cx, cy, hx, hy,
            fill="red", width=2, tags=tag
        )

    def display_status(self):
                # Pulizia terminale veloce
                #print("\033[H", end="")
                sigma = np.sqrt(np.diag(self.P))
                
                print("==================================================")
                print(f"       EKF DASHBOARD   ")
                print("==================================================")
                print(f"ACTUAL STATE (cm):")
                print(f" Position:  X={self.X[0,0]/100:.2f}m, Y={self.X[1,0]/100:.2f}m")
                print(f" Yaw:  θ={math.degrees(self.X[2,0]):.2f}°")
                print(f" Yaw rate:    ω={math.degrees(self.X[3,0]):.2f}°/s")
                print(f" Gyro Bias:  β={math.degrees(self.X[4,0]):.4f}°/s")
                print("--------------------------------------------------")
                print(f"Uncertain σ (cm):")
                print(f" σX:     {sigma[0]:.3f} cm")
                print(f" σY:     {sigma[1]:.3f} cm")
                print(f" σθ:     {math.degrees(sigma[2]):.3f} °")
        