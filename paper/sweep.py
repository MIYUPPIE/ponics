"""Move-penalty sweep for the paper: MPC vs GP-MPC, both suites, 10 seeds per cell.
Writes paper/data/sweep.json. Deterministic (seeded scenarios, fixed models).

  uv run python paper/sweep.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PENALTIES = (0.003, 0.01, 0.02, 0.05)
OUT = Path(__file__).parent / "data" / "sweep.json"


def one(args):
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    kind, r, suite, seed = args
    from ponics_bench.runner import run_episode
    from ponics_bench.scenarios import Scenario
    from ponics_control.gp_mpc import GpMpc
    from ponics_control.mpc import Mpc, MpcParams
    from ponics_eval.metrics import episode_metrics
    from ponics_eval.registry import LEARNED
    from ponics_learning.gp import ResidualGp

    p = MpcParams(r_du=(5e-4, r))
    c = Mpc(params=p) if kind == "mpc" else GpMpc(params=p, residual=ResidualGp.load(LEARNED["gp_mpc"]))
    return kind, r, suite, seed, episode_metrics(run_episode(c, Scenario(suite, seed)))


def main() -> None:
    jobs = [(k, r, s, sd) for k in ("mpc", "gp_mpc") for r in PENALTIES for s in ("train", "unseen") for sd in range(10)]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res = list(ex.map(one, jobs, chunksize=1))
    out = [{"controller": k, "r_du": r, "suite": s, "seed": sd, **m} for k, r, s, sd, m in res]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=0))
    print(f"wrote {OUT} ({len(out)} episodes)")


if __name__ == "__main__":
    main()
