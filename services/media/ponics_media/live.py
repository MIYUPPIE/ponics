"""ponics-live: run the simulator live in a desktop window.

  ponics-live                                    gp_mpc, train suite, seed 0
  ponics-live --controller pid --suite unseen --seed 3 --speed 48
  ponics-live --controller mpc_tight --days 10

The window shows the reservoir (tinted by true pH), probe readings, pumps, crop and the
pH/EC traces as the episode runs. --speed is simulated hours per second (the simulator
itself runs ~250 control steps/s, so up to ~60 h/s is real time). Close the window to stop;
the end-of-run metrics are printed.
"""

from __future__ import annotations

import argparse
import time

import matplotlib
import numpy as np

from ponics_bench.runner import observe
from ponics_bench.scenarios import SUITES, Scenario
from ponics_contracts import CONTROL_DT_S, DAY_S, Setpoints
from ponics_eval.registry import NAMES, make
from ponics_sensors import Probes
from ponics_sim.reservoir import NftSystem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ponics-live")
    ap.add_argument("--controller", choices=NAMES, default="gp_mpc")
    ap.add_argument("--suite", choices=SUITES, default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--speed", type=float, default=24.0, help="simulated hours per second")
    a = ap.parse_args(argv)
    try:
        matplotlib.use("TkAgg")
    except Exception:  # pragma: no cover - no display
        print("No GUI backend available; use `ponics-view` or `ponics-video` instead.")
        return 2
    import matplotlib.pyplot as plt

    from ponics_media.render import BG, CCOL, DESC, SUITE_TXT, Tank

    ctrl = make(a.controller)
    sc = Scenario(a.suite, a.seed, "sensor", a.days)
    sys_ = NftSystem(sc.config())
    probes = Probes(sc.sensor_seed)
    ctrl.reset()
    sp = Setpoints()
    steps = int(a.days * DAY_S / CONTROL_DT_S)

    plt.ion()
    fig = plt.figure(figsize=(13, 6.5), facecolor=BG)
    fig.canvas.manager.set_window_title(f"ponics live · {a.controller} · {a.suite} seed {a.seed}")
    tank = Tank(fig.add_axes([0.01, 0.12, 0.30, 0.78]), a.controller, big=False)
    ax_ph, ax_ec = fig.add_axes([0.37, 0.45, 0.60, 0.45]), fig.add_axes([0.37, 0.10, 0.60, 0.25])
    for ax, lo, hi, band, lab in ((ax_ph, 4, 8, (sp.ph_lo, sp.ph_hi), "true pH"),
                                  (ax_ec, 1.0, 3.5, (sp.ec - sp.ec_band, sp.ec + sp.ec_band), "EC dS/m")):
        ax.set_facecolor("#161e1b")
        ax.axhspan(*band, color="#1d3a2d")
        ax.set_xlim(0, a.days)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(lab)
        for d in sc.config().dump_days:
            if d < a.days:
                ax.axvline(d + 8 / 24, color="#8fa09a", ls="--", lw=0.8)
    (l_ph,) = ax_ph.plot([], [], color=CCOL.get(a.controller, "w"), lw=1.8)
    (l_ec,) = ax_ec.plot([], [], color=CCOL.get(a.controller, "w"), lw=1.8)
    ax_ec.set_xlabel("day")
    title = fig.text(0.02, 0.95, f"{a.controller} · {DESC.get(a.controller, '')} · {SUITE_TXT[a.suite]}", fontsize=13)

    hist = {k: [] for k in ("t", "ph", "ec", "ph_meas", "ab", "acid", "base", "fw", "stress")}
    t_start = time.perf_counter()
    for k in range(steps):
        if not plt.fignum_exists(fig.number):
            break
        obs = observe(sys_, probes, sp)
        dose = sys_.step(ctrl.act(obs))
        for key, v in (("t", sys_.t / DAY_S), ("ph", sys_.ph), ("ec", sys_.ec), ("ph_meas", obs.ph), ("ab", dose.ab_ml),
                       ("acid", dose.acid_ml), ("base", dose.base_ml), ("fw", sys_.fresh_weight_g()),
                       ("stress", sys_.last_rates.f_stress)):
            hist[key].append(v)
        if k % 4 == 3:  # redraw once per simulated hour
            r = {key: hist[key] for key in hist}
            tank.update({"ph": r["ph"], "ec": r["ec"], "ph_meas": r["ph_meas"], "ab": r["ab"], "acid": r["acid"],
                         "base": r["base"], "fw": r["fw"], "stress": r["stress"]}, len(r["ph"]) - 1)
            l_ph.set_data(hist["t"], hist["ph"])
            l_ec.set_data(hist["t"], hist["ec"])
            h = sys_.t / 3600
            title.set_text(f"{a.controller} · {SUITE_TXT[a.suite]} · day {int(h // 24)} {int(h % 24):02d}:00")
            wait = (k + 1) / 4 / a.speed - (time.perf_counter() - t_start)
            plt.pause(max(wait, 0.001))
    ph = np.array(hist["ph"])
    if len(ph):
        print(f"{a.controller} {a.suite} seed {a.seed}: {len(ph) / 96:.1f} days, pH in band "
              f"{np.mean((ph >= sp.ph_lo) & (ph <= sp.ph_hi)):.1%}, pH RMS {np.sqrt(np.mean((ph - sp.ph) ** 2)):.3f}, "
              f"fresh weight {hist['fw'][-1]:.0f} g/plant, acid {sum(hist['acid']) / 1000:.2f} L, base {sum(hist['base']) / 1000:.2f} L")
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
