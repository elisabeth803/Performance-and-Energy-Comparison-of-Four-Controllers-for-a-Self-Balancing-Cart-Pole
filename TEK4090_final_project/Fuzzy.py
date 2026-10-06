import numpy as np
import skfuzzy as fuzz
from   skfuzzy import control as ctrl

class Fuzzy:

    def __init__(self, u_max=15, phi_range=(-2, 2), phi_dot_range=(-8, 8)):
        self.dt          = 1/240
        self.u_max       = u_max
        self.kx          = 0.12
        self.kxd         = 1.2
        self.ki          = 0.005 
        self.phi_ref_max = np.deg2rad(5)

        # universes
        self.phi     = ctrl.Antecedent(np.linspace(phi_range[0], phi_range[1], 101), "phi")
        self.phi_dot = ctrl.Antecedent(np.linspace(phi_dot_range[0], phi_dot_range[1], 101), "phi_dot")
        self.force   = ctrl.Consequent(np.linspace(-u_max, u_max, 101), "force")

        # membership functions (symmetric)
        p = self.phi.universe
        self.phi["NL"] = fuzz.trapmf(p, [phi_range[0], phi_range[0], -0.4, -0.18]) # NL = Negative Large
        self.phi["NS"] = fuzz.trimf(p,  [-0.4, -0.16, 0])                          # NS = Negative Small
        self.phi["Z"]  = fuzz.trimf(p,  [-0.06, 0,    0.06])                       # Z = Zero
        self.phi["PS"] = fuzz.trimf(p,  [0,     0.16, 0.4])                        # PS = Positive Small
        self.phi["PL"] = fuzz.trapmf(p, [0.18, 0.4, phi_range[1], phi_range[1]])   # PL = Positive Large

        pd = self.phi_dot.universe
        self.phi_dot["NB"] = fuzz.trapmf(pd, [ phi_dot_range[0], phi_dot_range[0], -4, -1.5]) # NB = Negative Big
        self.phi_dot["NS"] = fuzz.trimf(pd,  [-4,    -1.2,  0])                               # NS = Negative Small
        self.phi_dot["Z"]  = fuzz.trimf(pd,  [-0.1,   0,    0.1])                             # Z = Zero
        self.phi_dot["PS"] = fuzz.trimf(pd,  [ 0,     1.2,  4])                               # PS = Positive Small
        self.phi_dot["PB"] = fuzz.trapmf(pd, [ 1.5, 4, phi_dot_range[1], phi_dot_range[1]])   # PB = Positive Big

        f = self.force.universe
        self.force["NL"] = fuzz.trapmf(f, [-u_max,     -u_max,    -0.8*u_max, -0.4*u_max])
        self.force["NS"] = fuzz.trimf(f,  [-0.8*u_max, -0.4*u_max, 0])
        self.force["Z"]  = fuzz.trimf(f,  [-0.1*u_max,  0,         0.1*u_max])
        self.force["PS"] = fuzz.trimf(f,  [ 0,          0.4*u_max, 0.8*u_max])
        self.force["PL"] = fuzz.trapmf(f, [ 0.4*u_max,  0.8*u_max, u_max,      u_max])
        
        # --- 5x5 PD-like rule table ---
        # rows: phi (NL, NS, Z, PS, PL)
        # cols: phi_dot (NB, NS, Z, PS, PB)
        phi_labels = ["NL", "NS", "Z", "PS", "PL"]
        pd_labels  = ["NB", "NS", "Z", "PS", "PB"]

        out = [
            ["NL", "NL", "NL", "NL", "NL"],  # phi NL
            ["NL", "NL", "NS", "Z",  "PS"],  # phi NS
            ["NL", "NS", "Z",  "PS", "PL"],  # phi Z 
            ["NS", "Z",  "PS", "PL", "PL"],  # phi PS
            ["PL",  "PL", "PL", "PL", "PL"], # phi PL
        ]

        rules = []
        for i, pl in enumerate(phi_labels):
            for j, dl in enumerate(pd_labels):
                rules.append(ctrl.Rule(self.phi[pl] & self.phi_dot[dl], self.force[out[i][j]])) # phi AND phi_dot -> force

        self._cs  = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(self._cs)

    def __call__(self, x):
        x_pos, x_vel, phi, phi_dot = map(float, x)
        xabs = abs(x_pos)        

        settled = (xabs < 0.05 and abs(x_vel) < 0.15 and abs(phi) < self.phi_ref_max and abs(phi_dot)< 0.6)

        # BANGBANG recovery-ish mode
        phi_switch = np.deg2rad(120)
        if abs(phi) > phi_switch:
            u = 0.8 * self.u_max * np.sign(phi_dot * np.cos(phi))
            return np.clip(u, -self.u_max, self.u_max)

        # Deadbands
        x_db    = 0.03 # meter 0.02-0.06
        xd_db   = 0.1 # m/s 0.05-0.15
        x_eff   = 0 if xabs < x_db else x_pos # If the cart is within 3 cm of center, treat position as 0.
        xd_eff  = 0 if abs(x_vel) < xd_db else x_vel # If speed is small, treat it as 0.
        phi_ref = 0 if settled else np.clip(-(self.kx * x_eff + self.kxd * xd_eff), -self.phi_ref_max, self.phi_ref_max) 
        
        phi_err = phi - phi_ref

        # clip inputs to universe
        phi_c = np.clip(phi_err, self.phi.universe[0], self.phi.universe[-1])
        pd_c  = np.clip(phi_dot, self.phi_dot.universe[0], self.phi_dot.universe[-1])

        self._sim = ctrl.ControlSystemSimulation(self._cs)

        try:
            self._sim.input["phi"] = phi_c
            self._sim.input["phi_dot"] = pd_c
            self._sim.compute()
            u_fuzzy = self._sim.output["force"]
        except Exception:
            u_fuzzy = 0

        # wall settings
        soft_zone = 1.5
        wall_k = 40  # 5-50, kick-wall
        wall_kv = 12 # damping into wall
        
        d = max(0, xabs - soft_zone)

        if d > 0:
            v_out = max(0, np.sign(x_pos)*x_vel)
            wall = -np.sign(x_pos) * (wall_k*d**2 + wall_kv*v_out)
        else:
            wall = 0
        
        headroom = max(0, self.u_max - abs(u_fuzzy))
        wall = np.clip(wall, -0.5*headroom, 0.5*headroom)

        u = np.clip(u_fuzzy + wall, -self.u_max, self.u_max)

        return u