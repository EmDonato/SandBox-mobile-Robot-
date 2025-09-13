import sympy as sp
import numpy as np

class Kalman:
    """
    Extended Kalman Filter (EKF) for differential-drive robot.
    State vector: [x, y, theta, vL, vR]^T
    Measurement:  [x_meas, y_meas, vL_meas, vR_meas]^T
    """

    def __init__(self, L=3.6, dt=0.1, xx=0.0, yy=0.0):
        self.L = L
        self.dt = dt

        # Symbolic variables
        x, y, theta, vL, vR, dt, L = sp.symbols('x y theta vL vR dt L')

        # Kinematic model
        x_next     = x + (vL + vR)/2 * sp.cos(theta) * dt
        y_next     = y + (vL + vR)/2 * sp.sin(theta) * dt
        theta_next = theta + (vR - vL)/L * dt
        vL_next    = vL
        vR_next    = vR

        f = sp.Matrix([x_next, y_next, theta_next, vL_next, vR_next])
        X = sp.Matrix([x, y, theta, vL, vR])

        # Jacobian
        Fsym = f.jacobian(X)

        # Numeric functions
        self.f_func = sp.lambdify((x, y, theta, vL, vR, dt, L), f, "numpy")
        self.F_func = sp.lambdify((x, y, theta, vL, vR, dt, L), Fsym, "numpy")

        # Initial state
        self.x_hat = np.zeros((5,1))
        self.x_hat[0] = xx
        self.x_hat[1] = yy
        self.P = np.eye(5) * 0.1

        # Covariances (consistent with robot noise)
        self.Q = np.diag([0.0025, 0.0025, 0.0025, 0.01, 0.01])
        self.R = np.diag([0.00625, 0.00625, 0.001, 0.001])

        # Observation matrix
        self.H = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
        ])

    def predict(self):
        """Prediction step."""
        x, y, theta, vL, vR = self.x_hat.flatten()
        self.x_hat = np.array(
            self.f_func(x, y, theta, vL, vR, self.dt, self.L),
            dtype=float
        ).reshape(5,1)
        Fk = np.array(self.F_func(x, y, theta, vL, vR, self.dt, self.L), dtype=float)
        self.P = Fk @ self.P @ Fk.T + self.Q

    def update(self, z):
        """Update step with measurement z = [x_meas, y_meas, vL_meas, vR_meas]."""
        z = np.array(z).reshape(-1,1)
        z_hat = self.H @ self.x_hat
        y_tilde = z - z_hat
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x_hat = self.x_hat + K @ y_tilde
        I = np.eye(5)
        self.P = (I - K @ self.H) @ self.P
