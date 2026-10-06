import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import argparse

from LQR import LQR
from MPC import MPC
from Fuzzy import Fuzzy
from HINF import HINF

from plotter import Plotter


class PendulumSim:
    """
    Cart-pole simulation using PyBullet.
    The controller is injected, so it can be swapped (LQR, PID, RL, etc.).
    """

    def __init__(self, controller=None, kick_magnitude=5, gui=True, dt=1/240, real_time=None):
        self.dt = dt
        self.sim_time = 0

        # Need to make it real time for visualization 
        self.real_time = gui if real_time is None else bool(real_time)

        # kick settings
        self.kick_time = 5.0             # when kick starts
        self.kick_duration = 0.1         # how long kick lasts
        self.kick_magnitude = kick_magnitude
        self.kick_end_time = 0.0
        self.current_kick = 0.0
        self.kick_has_happened = False   # ensures the kick happens only once(for testing)

        self.controller = controller

        # Logging arrays
        self.log_t = []
        self.log_x = []
        self.log_xdot = []
        self.log_phi = []
        self.log_phidot = []
        self.log_u_ctrl = []
        self.log_u_kick = []
        self.log_u_total = []

        self.log_e_u_ctrl = []
        self.log_e_mech_total = []

        self.e_u_ctrl = 0.0
        self.e_mech_total = 0.0

        self._dx = None

        #PyBullet setup
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF("plane.urdf")
        self.cartpole_id = p.loadURDF("cartPole.urdf")

        self.cart_joint = 0
        self.pole_joint = 1

        p.setJointMotorControl2(self.cartpole_id, self.cart_joint, controlMode=p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole_id, self.pole_joint, controlMode=p.VELOCITY_CONTROL, force=0)

        p.resetJointState(self.cartpole_id, self.pole_joint, targetValue=math.pi + math.radians(20))

    def _get_state(self):
        x_pos, x_vel, _, _ = p.getJointState(self.cartpole_id, self.cart_joint)
        theta, theta_dot, _, _ = p.getJointState(self.cartpole_id, self.pole_joint)

        phi = theta - math.pi
        phi = (phi + math.pi) % (2 * math.pi) - math.pi

        return np.array([x_pos, x_vel, phi, theta_dot], dtype=float)

    def _compute_control(self, x):
        if self.controller is None:
            return 0.0

        if callable(self.controller):
            try:
                u = self.controller(x, self.sim_time)
            except TypeError:
                u = self.controller(x)
            return float(u)

        if hasattr(self.controller, "compute_control"):
            return float(self.controller.compute_control(x, self.sim_time))

        return 0.0

    def _compute_kick_disturbance(self):
        #Apply a single kick once
        if (not self.kick_has_happened) and (self.sim_time >= self.kick_time):
            direction = 1 if random.random() > 0.5 else -1
            self.current_kick = direction * self.kick_magnitude
            self.kick_end_time = self.sim_time + self.kick_duration
            self.kick_has_happened = True
            print(f"Kick! F = {self.current_kick:.2f} N at t = {self.sim_time:.2f} s")

        if self.sim_time < self.kick_end_time:
            return self.current_kick

        return 0.0

    def _apply_control(self, force):
        p.setJointMotorControl2(
            self.cartpole_id,
            self.cart_joint,
            controlMode=p.TORQUE_CONTROL,
            force=force
        )

    def step(self):
        x = self._get_state()

        u_ctrl  = self._compute_control(x)
        u_kick  = self._compute_kick_disturbance()
        u_total = u_ctrl + u_kick

        self.e_u_ctrl += (u_ctrl ** 2) * self.dt

        if self._dx is None:
            dx = 0.0
        else:
            dx = x[0] - self._dx
        self._dx = x[0]

        self.e_mech_total += abs(u_total * dx)

        # Logging
        self.log_t.append(self.sim_time)
        self.log_x.append(x[0])
        self.log_xdot.append(x[1])
        self.log_phi.append(x[2])
        self.log_phidot.append(x[3])
        self.log_u_ctrl.append(u_ctrl)
        self.log_u_kick.append(u_kick)
        self.log_u_total.append(u_total)
        self.log_e_u_ctrl.append(self.e_u_ctrl)
        self.log_e_mech_total.append(self.e_mech_total)

        self._apply_control(u_total)

        p.stepSimulation()

        if self.real_time:
            time.sleep(self.dt)

        self.sim_time += self.dt

    def run(self):
        while True:
            self.step()

    def get_logs(self):
        #Export logs

        t = np.array(self.log_t)
        phi = np.array(self.log_phi)

        return {
            "t": t,
            "x": np.array(self.log_x),
            "xdot": np.array(self.log_xdot),
            "phi": phi,
            "phi_deg": np.rad2deg(phi),
            "phidot": np.array(self.log_phidot),
            "u_ctrl": np.array(self.log_u_ctrl),
            "u_kick": np.array(self.log_u_kick),
            "u_total": np.array(self.log_u_total),
            "e_u_ctrl": np.array(self.log_e_u_ctrl),
            "e_mech_total": np.array(self.log_e_mech_total),
        }

    @staticmethod
    def compute_metrics(logs: dict):
        # compute comparable metrics from logs.
        phi_deg = logs["phi_deg"]
        u_total = logs["u_total"]

        max_abs_phi = np.max(np.abs(phi_deg)) if len(phi_deg) else 0
        rms_phi = np.sqrt(np.mean(phi_deg ** 2)) if len(phi_deg) else 0
        rms_u = np.sqrt(np.mean(u_total ** 2)) if len(u_total) else 0

        e_u_ctrl_final = logs["e_u_ctrl"][-1] if len(logs["e_u_ctrl"]) else 0
        e_mech_total_final = logs["e_mech_total"][-1] if len(logs["e_mech_total"]) else 0

        return {
            "max_abs_phi_deg": max_abs_phi,
            "rms_phi_deg": rms_phi,
            "rms_u_total": rms_u,
            "e_u_ctrl_final": e_u_ctrl_final,
            "e_mech_total_final": e_mech_total_final,
        }

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass

    def run_for(self, T=10, plot=True, disconnect=True, title=None, save_energy_path="energy.png", return_logs=False):
        while self.sim_time < T:
            self.step()

        logs = self.get_logs()

        if plot:
            plotter = Plotter()
            if title is None:
                title = f"{type(self.controller).__name__} Simulation (first {T:.1f} seconds)"
            plotter.plot_all(logs, title=title, save_energy_path=save_energy_path)

        if disconnect:
            self.close()

        if return_logs:
            return logs
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cart-Pole PyBullet simulation")

    parser.add_argument(
        "--controller",
        type=str,
        default="mpc",
        choices=["none", "lqr", "mpc", "fuzzy", "hinf"],
        help="Which controller to use",
    )

    parser.add_argument("--T", type=float, default=10.0, help="Simulation time [s]")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI")

    parser.add_argument("--kick", type=float, default=15, help="Kick magnitude [N]")

    # Controller params
    parser.add_argument("--u-max", type=float, default=None, help="Control saturation u_max")
    parser.add_argument("--horizon", type=int, default=None, help="MPC horizon length")

    args = parser.parse_args()

    # GUI flag logic
    gui = True
    if args.gui:
        gui = True
    if args.no_gui:
        gui = False

    # Build controller from name
    controller = None
    if args.controller == "none":
        controller = None

    elif args.controller == "lqr":
        u_max = 20 if args.u_max is None else args.u_max
        controller = LQR(u_max=u_max)

    elif args.controller == "mpc":
        u_max = 20 if args.u_max is None else args.u_max
        horizon = 40 if args.horizon is None else args.horizon
        controller = MPC(horizon=horizon, u_max=u_max)

    elif args.controller == "fuzzy":
        u_max = 20 if args.u_max is None else args.u_max
        controller = Fuzzy(u_max=u_max)

    elif args.controller == "hinf":
        u_max = 15 if args.u_max is None else args.u_max
        controller = HINF(u_max=u_max)

    sim = PendulumSim(controller=controller, kick_magnitude=args.kick, gui=gui)
    sim.run_for(T=args.T)