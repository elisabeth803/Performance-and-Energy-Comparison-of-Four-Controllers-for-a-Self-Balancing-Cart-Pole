import numpy as np
import control as ct
class HINF:
    def __init__(self, dt=1.0 / 240.0, u_max=9.0, wb=2.0, wh=21, wl=3.8, wt1=22, wt2=3.7, fix_A=1):
        self.dt = dt
        self.u_max = u_max
        A, B, C, D = self._make_sys()

        if fix_A:
            A[0, 0] = -1e-4  

        P = ct.ss(A, B, C, D)

        Wp, Wu, Wt = self._make_weights(wb, wh, wl, wt1, wt2)
        K_cont, CL, tuple = ct.mixsyn(P, Wp, Wu, Wt)
        gamma, _ = tuple
        self.gamma = gamma

        K_cont = ct.ss(K_cont)
        K_disc = ct.sample_system(K_cont, self.dt, method="tustin")

        self.A_disc = K_disc.A
        self.B_disc = K_disc.B
        self.C_disc = K_disc.C
        self.D_disc = K_disc.D

        self.x_K_disc = np.zeros(self.A_disc.shape[0])

    def _make_sys(self):
        M = 0.5
        m = 0.5
        b = 0.15
        I = 0.0041667
        g = 9.81
        l = 0.5
        p = I*(M+m) + M*m*l**2

        A = np.array([[0, 1, 0, 0], [0, -(I + m*l**2)*b/p, (m**2*g*l**2)/p, 0], 
                      [0, 0, 0, 1], [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0]])

        B = np.array([0, (I + m*l**2)/p, 0, m*l/p]).reshape(-1, 1)  # (4,1)

        C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        D = np.array([[0], [0]])

        return A, B, C, D

    def _make_weights(self, wb, wh, wl, wt1, wt2):
        s = ct.TransferFunction.s

        Wp = (s / wb + 1) / (s / (wb / 20) + 1)
        Wu = (s / wl + 1) / (s / wh + 1)
        Wt = (s / wt1 + 1 + 1e-3) / (s / wt2 + 1)

        Wp2 = ct.append(1e-6 * Wp, 1.1 * Wp)  
        Wt2 = ct.append(1.1 * Wt, 1.1 * Wt)   
        Wu1 = 1 * Wu                      

        return Wp2, Wu1, Wt2


    def __call__(self, x, t=None):
        x = np.asarray(x).flatten()
        pos = x[0]
        theta = x[2]
        phi = theta - np.pi             
        phi = (phi + np.pi) % (2*np.pi) - np.pi   # wrap to [-pi, pi]
        y = np.array([pos, phi])      

        u_calc = (self.C_disc @ self.x_K_disc + self.D_disc @ y).item()

        u_calc = -float(u_calc)  
        u_clipped = float(np.clip(u_calc, -self.u_max, self.u_max))
        saturated = 0
        if (abs(u_calc) > self.u_max) and (np.sign(u_calc) == np.sign(u_clipped)):      
          saturated = 1

        if not saturated:
            self.x_K_disc = self.A_disc @ self.x_K_disc + self.B_disc @ y
        else:
            pass    
        return u_clipped