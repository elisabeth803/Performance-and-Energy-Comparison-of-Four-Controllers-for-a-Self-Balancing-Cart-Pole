import numpy as np
from scipy.optimize import minimize
from numba import njit

@njit
def dynamics_numba(x, u, M, m, l, g):
    xdot   = x[1]
    phi    = x[2]
    phidot = x[3]

    force = u
    total_mass = M + m
    polemass_length = m * l

    temp = (force + polemass_length * phidot * phidot * np.sin(phi)) / total_mass
    phiacc = (g * np.sin(phi) - np.cos(phi) * temp) / (l * (4.0 / 3.0 - m * np.cos(phi) * np.cos(phi) / total_mass))
    xacc = temp - polemass_length * phiacc * np.cos(phi) / total_mass

    return np.array([xdot, xacc, phidot, phiacc], dtype=np.float64)


@njit
def rk4_step_numba(x, u, dt, M, m, l, g):
    k1 = dynamics_numba(x, u, M, m, l, g)
    k2 = dynamics_numba(x + 0.5 * dt * k1, u, M, m, l, g)
    k3 = dynamics_numba(x + 0.5 * dt * k2, u, M, m, l, g)
    k4 = dynamics_numba(x + dt * k3, u, M, m, l, g)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit
def simulate_and_cost_numba(u_seq, x0, x_ref, Q, R_scalar, Qf,
                            dt, N, M, m, l, g):
    x = x0.copy()
    cost = 0.0

    x_min = -2.0
    x_max =  2.0
    soft_zone = 1.5       
    wall_penalty = 1e4

    for k in range(N):
        u = u_seq[k]
        dx = x - x_ref

        cost += dx @ (Q @ dx) + R_scalar * u * u

        x_cart = x[0]

        if x_cart > soft_zone:
            d = x_cart - soft_zone
            cost += wall_penalty * d * d
        elif x_cart < -soft_zone:
            d = -soft_zone - x_cart
            cost += wall_penalty * d * d

        x = rk4_step_numba(x, u, dt, M, m, l, g)

    dx = x - x_ref
    cost += dx @ (Qf @ dx)

    x_cart = x[0]
    if x_cart > x_max:
        d = x_cart - x_max
        cost += wall_penalty * 10.0 * d * d
    elif x_cart < x_min:
        d = x_min - x_cart
        cost += wall_penalty * 10.0 * d * d

    return cost

class MPC:

    def __init__(self, dt=0.02, horizon=20, u_max=5.0, x_ref=None):
        self.M = 0.5   
        self.m = 0.5   
        self.l = 0.5   
        self.g = 9.81  

        self.dt = float(dt)
        self.N = int(horizon)
        self.u_max = float(u_max)

        self.Q = np.diag(np.array([10.0, 0.1, 40.0, 1.0]))
        self.R_scalar = 0.01
        self.Qf = self.Q.copy()

        if x_ref == None:
            x_ref = np.zeros(4) 
        else:
            x_ref = np.asarray(x_ref)
        self.x_ref = x_ref

        self.u_seq = np.zeros(self.N)
        self.bounds = [(-self.u_max, self.u_max)] * self.N


    def _simulate_and_cost(self, u_seq, x0):
        return simulate_and_cost_numba(
            u_seq, x0,self.x_ref, self.Q, self.R_scalar, self.Qf, self.dt, self.N, self.M, self.m, self.l, self.g,
        )

    def __call__(self, x, t=None):
        x0 = np.asarray(x).reshape(4,)

        def objective(u_flat):
            return self._simulate_and_cost(u_flat, x0)

        u0 = self.u_seq.copy()
        if self.N > 1:
            u0[:-1] = u0[1:]
        u0[-1] = 0.0

        res = minimize(objective, u0, method="L-BFGS-B", bounds=self.bounds, options={"maxiter": 50, "ftol": 1e-4},)

        if res.success:
            u_opt = res.x
        else:
            u_opt = u0

        self.u_seq = u_opt
        u = float(self.u_seq[0])
        if u > self.u_max:
            u = self.u_max
        elif u < -self.u_max:
            u = -self.u_max

        return u