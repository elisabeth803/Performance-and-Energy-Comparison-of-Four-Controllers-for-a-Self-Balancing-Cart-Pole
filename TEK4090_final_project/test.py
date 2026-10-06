import os
import csv
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from pendulumSim import PendulumSim

from LQR import LQR
from MPC import MPC
from Fuzzy import Fuzzy
from HINF import HINF

from plotter import Plotter


def make_controllers(u_max: float, horizon: int):
    controllers = {
        "none": None,
        "lqr": LQR(u_max=u_max),
        "mpc": MPC(horizon=horizon, u_max=u_max),
        "fuzzy": Fuzzy(u_max=u_max),
        "hinf": HINF(u_max=u_max),
    }
    return controllers 


def compute_metrics(logs: dict):
    metrics = PendulumSim.compute_metrics(logs)

    st, _ = Plotter.compute_settling_time(
        t=logs["t"],
        x=logs["x"],
        phi_rad=logs["phi"],
        x_ref=0.0,
        phi_ref=0.0,
        epsilon=0.05,  # 5%
        phi_tol=np.deg2rad(2),
        settle_hold=0.5,
    )
    metrics["settling_time_s"] = float(st) if st is not None else np.nan
    return metrics


def shade_kicks(ax, logs: dict):
    intervals = Plotter.get_kick_intervals(logs["t"], logs["u_kick"])
    for (t0, t1) in intervals:
        ax.axvspan(t0, t1, alpha=0.15)


def run_one(controller_name, controller_obj, u_max, u_kick, T, dt, seed, out_logs_dir):
    random.seed(seed)
    np.random.seed(seed)

    sim = PendulumSim(controller=controller_obj, kick_magnitude=u_kick, gui=False, dt=dt)

    logs = sim.run_for(T=T, plot=False, disconnect=True, return_logs=True)

    metrics = compute_metrics(logs)

    tag = f"{controller_name}__umax{int(u_max)}__ukick{int(u_kick)}"
    npz_path = os.path.join(out_logs_dir, f"{tag}.npz")
    np.savez_compressed(npz_path, **logs)

    row = {
        "controller": controller_name,
        "u_max": float(u_max),
        "u_kick": float(u_kick),
        "seed": int(seed),
        "npz_path": npz_path,
        **metrics,
    }
    return logs, row


def write_summary_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_overlays_by_controller(results, controllers, u_max_list, u_kick, out_plots_dir):
    """
    One figure per kick:
      - rows = controllers
      - each subplot overlays u_max curves
    Makes:
      phi overlays + u_total overlays + x_dot overlays
    """
    # --- phi(t) grid ---
    fig, axs = plt.subplots(len(controllers), 1, sharex=True, figsize=(12, 2.2 * len(controllers)))
    if len(controllers) == 1:
        axs = [axs]
    fig.suptitle(f"phi(t) overlays vs u_max  |  u_kick={u_kick}")

    for i, cname in enumerate(controllers):
        ax = axs[i]
        for u_max in u_max_list:
            logs = results[(cname, u_max, u_kick)]
            ax.plot(logs["t"], logs["phi_deg"], label=f"u_max={u_max}")
        shade_kicks(ax, results[(cname, u_max_list[0], u_kick)])
        ax.set_ylabel(f"{cname}\nphi [deg]")
        ax.grid(True)
        ax.legend(ncol=min(4, len(u_max_list)), fontsize=9)

    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(out_plots_dir, f"phi_grid__ukick{int(u_kick)}.png"), dpi=160)
    plt.close(fig)

    # --- u_total(t) grid ---
    fig, axs = plt.subplots(len(controllers), 1, sharex=True, figsize=(12, 2.2 * len(controllers)))
    if len(controllers) == 1:
        axs = [axs]
    fig.suptitle(f"u_total(t) overlays vs u_max  |  u_kick={u_kick}")

    for i, cname in enumerate(controllers):
        ax = axs[i]
        for u_max in u_max_list:
            logs = results[(cname, u_max, u_kick)]
            ax.plot(logs["t"], logs["u_total"], label=f"u_max={u_max}")
        shade_kicks(ax, results[(cname, u_max_list[0], u_kick)])
        ax.set_ylabel(f"{cname}\nu [N]")
        ax.grid(True)
        ax.legend(ncol=min(4, len(u_max_list)), fontsize=9)

    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(out_plots_dir, f"utotal_grid__ukick{int(u_kick)}.png"), dpi=160)
    plt.close(fig)

    # --- x_dot(t) grid ---
    fig, axs = plt.subplots(len(controllers), 1, sharex=True, figsize=(12, 2.2 * len(controllers)))
    if len(controllers) == 1:
        axs = [axs]
    fig.suptitle(f"x_dot(t) overlays vs u_max  |  u_kick={u_kick}")

    def get_xdot(logs):
        # Prefer a logged velocity if available, otherwise compute from x(t)
        if "x_dot" in logs:
            return logs["x_dot"]
        if "xdot" in logs:
            return logs["xdot"]
        if "x_dot_mps" in logs:
            return logs["x_dot_mps"]
        return np.gradient(logs["x"], logs["t"])

    for i, cname in enumerate(controllers):
        ax = axs[i]
        for u_max in u_max_list:
            logs = results[(cname, u_max, u_kick)]
            xdot = get_xdot(logs)
            ax.plot(logs["t"], xdot, label=f"u_max={u_max}")
        shade_kicks(ax, results[(cname, u_max_list[0], u_kick)])
        ax.set_ylabel(f"{cname}\nx_dot [m/s]")
        ax.grid(True)
        ax.legend(ncol=min(4, len(u_max_list)), fontsize=9)

    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(out_plots_dir, f"xdot_grid__ukick{int(u_kick)}.png"), dpi=160)
    plt.close(fig)


def plot_summary_curves(summary_rows, controllers, u_max_list, u_kick_list, out_plots_dir):
    """
    For each kick:
      - plot max|phi| vs u_max for all controllers
      - plot settling_time vs u_max for all controllers
      - plot final energy vs u_max for all controllers
    """
    def rows_for(cname, u_kick):
        return [r for r in summary_rows if r["controller"] == cname and r["u_kick"] == float(u_kick)]

    for u_kick in u_kick_list:
        # max|phi| vs u_max
        fig = plt.figure(figsize=(10, 6))
        for cname in controllers:
            rs = rows_for(cname, u_kick)
            rs = sorted(rs, key=lambda r: r["u_max"])
            xs = [r["u_max"] for r in rs]
            ys = [r["max_abs_phi_deg"] for r in rs]
            plt.plot(xs, ys, marker="o", label=cname)
        plt.title(f"Peak angle vs u_max  |  u_kick={u_kick}")
        plt.xlabel("u_max")
        plt.ylabel("max |phi| [deg]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_plots_dir, f"summary_maxphi__ukick{int(u_kick)}.png"), dpi=160)
        plt.close(fig)

        # settling time vs u_max
        fig = plt.figure(figsize=(10, 6))
        for cname in controllers:
            rs = rows_for(cname, u_kick)
            rs = sorted(rs, key=lambda r: r["u_max"])
            xs = [r["u_max"] for r in rs]
            ys = [r["settling_time_s"] for r in rs]
            plt.plot(xs, ys, marker="o", label=cname)
        plt.title(f"Settling time vs u_max  |  u_kick={u_kick}  (NaN = did not settle)")
        plt.xlabel("u_max")
        plt.ylabel("settling time [s]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_plots_dir, f"summary_settling__ukick{int(u_kick)}.png"), dpi=160)
        plt.close(fig)

        # energy vs u_max
        fig = plt.figure(figsize=(10, 6))
        for cname in controllers:
            rs = rows_for(cname, u_kick)
            rs = sorted(rs, key=lambda r: r["u_max"])
            xs = [r["u_max"] for r in rs]
            ys = [r["e_u_total_final"] for r in rs]
            plt.plot(xs, ys, marker="o", label=cname)
        plt.title(f"Cumulative total energy vs u_max  |  u_kick={u_kick}")
        plt.xlabel("u_max")
        plt.ylabel("E_u_total(final)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_plots_dir, f"summary_energy__ukick{int(u_kick)}.png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep all controllers over u_max and u_kick.")
    parser.add_argument("--T", type=float, default=10.0, help="Simulation time [s]")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Timestep [s]")
    parser.add_argument("--horizon", type=int, default=40, help="MPC horizon (for MPC controllers)")
    parser.add_argument("--out", type=str, default="sweep_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (same disturbance pattern each run)")
    args = parser.parse_args()

    u_max_list = [5, 10, 15, 20]
    u_kick_list = [5, 10, 15]

    out_dir = args.out
    out_logs_dir = os.path.join(out_dir, "logs")
    out_plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_logs_dir, exist_ok=True)
    os.makedirs(out_plots_dir, exist_ok=True)

    # Determine controller set once (names only), but instantiate per u_max
    controller_names = list(make_controllers(u_max=u_max_list[0], horizon=args.horizon).keys())

    results = {}       # (controller, u_max, u_kick) -> logs
    summary_rows = []  # list of dicts

    for u_kick in u_kick_list:
        for u_max in u_max_list:
            controllers = make_controllers(u_max=u_max, horizon=args.horizon)
            for cname in controller_names:
                cobj = controllers[cname]
                # Keep disturbance directions identical across ALL runs:
                # use same seed for every run (so "Kick!" sign pattern is the same everywhere)
                logs, row = run_one(
                    controller_name=cname,
                    controller_obj=cobj,
                    u_max=u_max,
                    u_kick=u_kick,
                    T=args.T,
                    dt=args.dt,
                    seed=args.seed,
                    out_logs_dir=out_logs_dir,
                )
                results[(cname, u_max, u_kick)] = logs
                summary_rows.append(row)

                print(
                    f"[DONE] {cname:5s}  u_max={u_max:>2}  u_kick={u_kick:>2} | "
                    f"max|phi|={row['max_abs_phi_deg']:.2f}deg  "
                    f"settle={row['settling_time_s'] if not np.isnan(row['settling_time_s']) else 'NaN'}"
                )

    # Save CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    write_summary_csv(csv_path, summary_rows)

    # Plots
    for u_kick in u_kick_list:
        plot_overlays_by_controller(results, controller_names, u_max_list, u_kick, out_plots_dir)
    plot_summary_curves(summary_rows, controller_names, u_max_list, u_kick_list, out_plots_dir)

    print("\nSaved:")
    print(f"  Logs:   {out_logs_dir}")
    print(f"  Plots:  {out_plots_dir}")
    print(f"  CSV:    {csv_path}\n")


if __name__ == "__main__":
    main()
