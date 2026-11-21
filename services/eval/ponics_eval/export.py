"""ponics-replay: record seed-0 episodes of each controller on both suites, decimated to
one sample per hour, for the replay viewer. Writes results/replay/replay.json and copies
the viewer page (ponics_eval/viewer/index.html) beside it; open it through any static
server, e.g. `python -m http.server -d results/replay`.

  ponics-replay                                  all controllers whose models exist
  ponics-replay --controllers mpc,gp_mpc_tight   a subset
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ponics_bench.runner import run_episode
from ponics_bench.scenarios import Scenario
from ponics_contracts import Setpoints
from ponics_eval.metrics import episode_metrics
from ponics_eval.registry import LEARNED, NAMES, make

ROOT = Path(__file__).resolve().parents[3]
PER_HOUR = 4  # control steps per hour


def _hourly(a: np.ndarray, how: str) -> list:
    a = a[: len(a) // PER_HOUR * PER_HOUR].reshape(-1, PER_HOUR)
    v = a.sum(1) if how == "sum" else a[:, 0]
    return [round(float(x), 3) for x in v]


def record(args: tuple[str, str]) -> tuple[str, str, dict]:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    name, suite = args
    sc = Scenario(suite, 0)
    log = run_episode(make(name), sc)
    a = log.arrays()
    cfg = sc.config()
    return name, suite, {
        "ph": _hourly(a["ph"], "first"), "ec": _hourly(a["ec"], "first"),
        "ph_meas": _hourly(a["ph_meas"], "first"), "ec_meas": _hourly(a["ec_meas"], "first"),
        "acid": _hourly(a["acid_ml"], "sum"), "base": _hourly(a["base_ml"], "sum"), "ab": _hourly(a["ab_ml"], "sum"),
        "fw": _hourly(a["fw"], "first"), "stress": _hourly(a["stress"], "first"),
        "metrics": {k: round(v, 4) for k, v in episode_metrics(log).items()},
        "dump_days": list(cfg.dump_days),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ponics-replay")
    ap.add_argument("--controllers", default=",".join(n for n in NAMES if n not in LEARNED or LEARNED[n].exists()))
    ap.add_argument("--out", default=str(ROOT / "results" / "replay" / "replay.json"))
    a = ap.parse_args(argv)
    names = a.controllers.split(",")
    jobs = [(n, s) for n in names for s in ("train", "unseen")]
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, (os.cpu_count() or 2) - 1))) as ex:
        done = list(ex.map(record, jobs))
    sp = Setpoints()
    out = {"setpoints": {"ec": sp.ec, "ph": sp.ph, "ph_lo": sp.ph_lo, "ph_hi": sp.ph_hi, "ec_band": sp.ec_band},
           "hours": len(done[0][2]["ph"]), "controllers": names,
           "suites": {"train": "soft tap water, 22/18 C room", "unseen": "hard tap water, 28/22 C dry room"},
           "runs": {s: {n: r for n, s2, r in done if s2 == s} for s in ("train", "unseen")}}
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    (path.parent / "index.html").write_text((Path(__file__).parent / "viewer" / "index.html").read_text())
    print(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(jobs)} episodes) and index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
