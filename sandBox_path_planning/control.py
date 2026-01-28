import math


def purePursuit(v_des, L, robot, path):
    """
    Pure Pursuit controller for trajectory tracking.

    The controller assumes that the robot follows a circular arc that:
    - starts at the robot position,
    - is tangent to the robot heading,
    - passes through a lookahead point on the path.

    The curvature of this arc is computed geometrically and converted
    into an angular velocity command.

    Args:
        v_des (float): Desired linear velocity [m/s].
        L (float): Lookahead distance [m].
        robot (object): Robot pose with attributes (x, y, theta).
        path (list of tuple): Reference path points [(x, y), ...].

    Returns:
        (float, float): Linear velocity v and angular velocity w.
    """
    if not path:
        return 0.0, robot.theta  # No target available

    # Target point
    xdes, ydes = path[0]
    x, y, theta = robot.x, robot.y, robot.theta

    # Transform target point into the robot's local frame
    dx = xdes - x
    dy = ydes - y
    x_r =  math.cos(theta) * dx + math.sin(theta) * dy
    y_r = -math.sin(theta) * dx + math.cos(theta) * dy

    # Curvature computation (avoid division by zero if L ≈ 0)
    if abs(L) < 1e-6:
        kappa = 0.0
    else:
        kappa = 2.0 * y_r / (L * L)

    # Angular velocity from curvature
    w = v_des * kappa

    # Return control commands (linear, angular velocities)
    return v_des, w



class PID:
    """
    Proportional-Integral-Derivative (PID) controller.
    
    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        integral (float): Accumulated integral error term.
        prev_error (float): Previous error value (for derivative computation).
    """

    def __init__(self, kp, ki, kd):
        """Initialize PID gains and internal state."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """
        Compute PID control output.
        
        Args:
            error (float): Current error signal.
            dt (float): Time step [s].
        
        Returns:
            float: Control action (u).
        """
        # Proportional term
        P = self.kp * error

        # Integral term (accumulated error)
        self.integral += error * dt
        I = self.ki * self.integral

        # Derivative term (error rate of change)
        D = self.kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        # Total control action
        return P + I + D
