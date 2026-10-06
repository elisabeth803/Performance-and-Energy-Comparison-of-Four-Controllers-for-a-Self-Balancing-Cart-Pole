import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    """
    Handles all plotting for the cart-pole simulation.
    Expects a logs dict (numpy arrays) from the simulator.
    """

    @staticmethod
    def get_kick_intervals(t, u_kick):
        """
        Return a list of (t_start, t_end) intervals when a kick is active,
        based on u_kick != 0.
        """
        if t is None or u_kick is None or len(t) == 0 or len(u_kick) == 0:
            return []

        t = np.asarray(t)
        u_kick = np.asarray(u_kick)

        active = u_kick != 0
        intervals = []

        in_kick = False
        start_time = None

        for i, is_active in enumerate(active):
            if is_active and not in_kick:
                in_kick = True
                start_time = t[i]
            elif (not is_active) and in_kick:
                in_kick = False
                end_time = t[i]
                intervals.append((start_time, end_time))

        if in_kick and start_time is not None:
            intervals.append((start_time, t[-1]))

        return intervals

    @staticmethod
    def _shade_intervals(axs, intervals):
        """
        Plots a shaded interval for the u_kick
        """
        if not intervals:
            return
        for (t_start, t_end) in intervals:
            for ax in axs:
                ax.axvspan(t_start, t_end, color="red", alpha=0.15)

    @staticmethod
    def compute_settling_time(t, x, phi_deg, kick_done, phi_ref=0, phi_tol=2, settle_hold=0.5):
        """
        Settling time = first time the signals stay inside the band continuously
        for at least `settle_hold` seconds.

        Returns: (settling_time, settling_idx)
        """

        if len(t) < 2:
            return None, None

        in_band_time = 0

        for i in range(len(t)):
            dt = (t[i] - t[i - 1]) if i > 0 else 0
            if t[i] > kick_done:

                in_band = (abs(phi_deg[i] - phi_ref) <= phi_tol)
                # print(abs(phi_deg[i] - phi_ref))

                if in_band:
                    if in_band_time >= settle_hold:
                        start_settling = t[i] - in_band_time + dt
                        settling_time  = t[i]
                        start_idx = int(np.searchsorted(t, start_settling))
                        settling_idx  = int(np.searchsorted(t, settling_time))
                        return start_settling, settling_time, start_idx, settling_idx
                    in_band_time += dt
                else:
                    in_band_time = 0

        return None, None, None, None

    def plot_settling_time(self, logs, phi_ref=0, phi_tol=2,
                           settle_hold=2,
                           title="Settling time"):
        """
        Plots x(t) and phi(t) with tolerance bands and marks the settling time (if found).
        Expects logs to contain: "t", "x", and either "phi" (rad) or "phi_deg" (deg).
        """
        t = logs["t"]
        x = logs["x"]
        u_kick = logs["u_kick"]
        phi_deg = logs["phi_deg"]
        
        kick_intervals = self.get_kick_intervals(t, u_kick)
        kick_done = kick_intervals[0][1]

        start_settling, settling_time, start_idx, settling_idx = self.compute_settling_time(
            t, x, phi_deg, kick_done, phi_tol=phi_tol,
            settle_hold=settle_hold
        )

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 3))
        fig.suptitle(title, y=1.2)

        # phi band (in degrees for readability)
        phi_ref_deg = phi_ref
        phi_tol_deg = phi_tol

        ax.plot(t, phi_deg, label="phi(t)")
        ax.axhspan(phi_ref_deg - phi_tol_deg, phi_ref_deg + phi_tol_deg, alpha=0.15, label="±phi_tol")
        ax.set_ylabel("phi (deg)")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()

        # mark settling time
        if start_settling is not None and settling_time is not None and settling_idx is not None:
            idx = min(settling_idx, len(t) - 1)
            t_set = t[idx]

            ax.axvline(t_set, linestyle="--")

            ax.plot(t_set, phi_deg[idx], "o")

            start_idx = min(start_idx, len(t) - 1)
            start_t_set = t[start_idx]

            ax.axvline(start_t_set, linestyle="--")

            ax.plot(start_t_set, phi_deg[start_idx], "o")

            ax.set_title(f"Settled at t = {t_set - 5:.2f} s (hold {settle_hold} s)")
        else:
            ax.set_title("Did not settle within simulation time")

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.subplots_adjust(top=0.85)
        plt.show()

        return settling_time, settling_idx

    def plot_state_and_control(self, logs, title="Cart-Pole Simulation"):
        t = logs["t"]
        x = logs["x"]
        xdot = logs["xdot"]
        phi_deg = logs["phi_deg"]
        u_ctrl = logs["u_ctrl"]
        u_total = logs["u_total"]
        u_kick = logs["u_kick"]

        kick_intervals = self.get_kick_intervals(t, u_kick)

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        fig.suptitle(title)

        axs[0].plot(t, x)
        axs[0].set_ylabel("x (m)")
        axs[0].set_title("Cart position")
        axs[0].grid(True)

        axs[1].plot(t, phi_deg)
        axs[1].set_ylabel("phi (deg)")
        axs[1].set_title("Pole angle")
        axs[1].grid(True)

        axs[2].plot(t, xdot)
        axs[2].set_ylabel("xdot (m/s)")
        axs[2].set_title("Cart velocity")
        axs[2].grid(True)

        axs[3].plot(t, u_total, label="u_total")
        axs[3].plot(t, u_ctrl, linestyle="--", label="u_ctrl")
        axs[3].set_ylabel("Force (N)")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_title("Control force")
        axs[3].legend()
        axs[3].grid(True)

        self._shade_intervals(axs, kick_intervals)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def plot_energy(self, logs, save_path="energy.png"):
        t = logs["t"]
        e_u_ctrl = logs["e_u_ctrl"]
        e_mech_total = logs["e_mech_total"]
        u_kick = logs["u_kick"]

        kick_intervals = self.get_kick_intervals(t, u_kick)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        fig.suptitle("Energy")

        axs[0].plot(t, e_u_ctrl)
        axs[0].set_title("Cumulative control energy")
        axs[0].grid(True)

        axs[1].plot(t, e_mech_total)
        axs[1].set_title("Cumulative mechanical energy")
        axs[1].grid(True)

        self._shade_intervals(axs, kick_intervals)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def plot_all(self, logs, title="Cart-Pole Simulation", save_energy_path="energy.png"):
        self.plot_state_and_control(logs, title=title)
        self.plot_energy(logs, save_path=save_energy_path)
        self.plot_settling_time(logs, title="Settling time HINF")