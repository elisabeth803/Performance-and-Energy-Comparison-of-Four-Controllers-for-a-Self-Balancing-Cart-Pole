import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
class LQR:
    def __init__(self, dt=1/240, u_max=5.0):
        self.dt = float(dt)

        M = 0.5      
        m = 0.5        
        l = 0.5        
        I = 0.0417167      
        g = 9.81

        D = I * (M + m) + M * m * l * l

        A = np.array([[0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, -g * l * l * m * m / D, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, (l * m * (M + m) * g) / D, 0.0] ])

        B = np.array([[0.0], [(I + m * l * l) / D], [0.0], [-(l * m) / D]])

        self.Ad, self.Bd, _, _, _ = cont2discrete((A, B, np.eye(A.shape[0]), 
                                                    np.zeros((A.shape[0], B.shape[1]))), 
                                                    self.dt, method="zoh")

        Q = np.diag([1.0, 0.1, 20.0, 1.0])
        R = np.array([[0.1]])
        self.Q_dt = Q * self.dt
        self.R_dt = R * self.dt

        self.K = self._dlqr() 
        self.u_max = float(u_max)

    def _dlqr(self):
        P = solve_discrete_are(self.Ad, self.Bd, self.Q_dt, self.R_dt)
        S = self.Bd.T @ P @ self.Bd + self.R_dt
        K = np.linalg.inv(S) @ (self.Bd.T @ P @ self.Ad)
        return K

    def __call__(self, x, t=None):
        x = np.asarray(x).reshape(-1)
        u = float(-self.K @ x)
        u = max(min(u, self.u_max), -self.u_max) 
        return u