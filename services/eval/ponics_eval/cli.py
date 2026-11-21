"""ponics-eval: run the seeded scenario grid, write results, check thresholds.

  ponics-eval                          all controllers, 10 seeds x 2 suites, sensor feedback
  ponics-eval --check                  also fail on regression vs evals/baseline.json or any
                                       break of evals/requirements.json
  ponics-eval --record                 overwrite evals/baseline.json with this run (deliberate act)
  ponics-eval --controllers mpc,pid --seeds 3       quick partial run
  ponics-eval --feedback sensor,truth  add truth-feedback runs (control error without probe error)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ponics_bench.runner import run_episode
from ponics_bench.scenarios import Scenario, grid
from ponics_eval.metrics import HIGHER_IS_BETTER, LOWER_IS_BETTER, aggregate, episode_metrics
from ponics_eval.registry import NAMES, make
from ponics_eval.report import plot_examples, write_summary

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "evals" / "baseline.json"
REQUIREMENTS = ROOT / "evals" / "requirements.json"
REL_TOL, ABS_TOL = 0.10, 0.005
MAX_SOLVE_MS = 200.0


def _single_thread() -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


_CACHE: dict = {}


def _one(args: tuple[str, Scenario]) -> tuple[str, Scenario, dict]:
    name, sc = args
    if name not in _CACHE:  # one controller object per worker; reset() runs per episode
        _CACHE[name] = make(name)
    return name, sc, episode_metrics(run_episode(_CACHE[name], sc))


def run_grid(controllers: list[str], seeds: int, feedback: list[str], workers: int) -> dict:
    jobs = [(c, sc) for c in controllers for sc in grid(seeds=range(seeds), feedback=tuple(feedback))]
    with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread) as ex:
        done = list(ex.map(_one, jobs, chunksize=1))
    groups: dict[str, list[dict]] = {}
    for name, sc, m in done:
        groups.setdefault(f"{name}|{sc.suite}|{sc.feedback}", []).append(m)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def check(results: dict, baseline: dict, requirements: dict) -> list[str]:
    fails = []
    for key, req in requirements.items():
        got = results.get(key)
        if got is None:
            continue
        for metric, limit in req.items():
            v = got[metric]
            bad = v > limit if metric in LOWER_IS_BETTER else v < limit
            if bad:
                fails.append(f"REQUIREMENT {key} {metric} = {v:.4g} (limit {limit})")
    for key, base in baseline.items():
        got = results.get(key)
        if got is None:
            continue
        for metric in LOWER_IS_BETTER:
            if got[metric] > base[metric] * (1 + REL_TOL) + ABS_TOL:
                fails.append(f"REGRESSION {key} {metric} {base[metric]:.4g} -> {got[metric]:.4g}")
        for metric in HIGHER_IS_BETTER:
            if got[metric] < base[metric] * (1 - REL_TOL) - ABS_TOL:
                fails.append(f"REGRESSION {key} {metric} {base[metric]:.4g} -> {got[metric]:.4g}")
    for key, got in results.items():
        if got["solve_p99_ms"] > MAX_SOLVE_MS:
            fails.append(f"SLOW {key} p99 solve {got['solve_p99_ms']:.1f} ms > {MAX_SOLVE_MS}")
    return fails


def examples(controllers: list[str], out: Path) -> None:
    for suite in ("train", "unseen"):
        logs = {c: run_episode(make(c), Scenario(suite, 0)) for c in controllers}
        plot_examples(logs, out / f"example_{suite}.png", f"{suite} suite, seed 0, sensor feedback")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ponics-eval")
    ap.add_argument("--controllers", default=",".join(NAMES))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--feedback", default="sensor")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(ROOT / "results" / "latest"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    controllers = a.controllers.split(",")
    full = set(controllers) == set(NAMES) and a.seeds == 10 and a.feedback == "sensor"
    if a.record and not full:
        print("--record needs the full default run", file=sys.stderr)
        return 2
    t0 = time.time()
    results = run_grid(controllers, a.seeds, a.feedback.split(","), a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    header = f"# ponics eval\n\n{a.seeds} seeds per cell, 35-day cycles, true pH/EC scored. Run took {time.time() - t0:.0f} s."
    write_summary(results, out / "summary.md", header)
    if not a.no_plots:
        examples(controllers, out)
    print((out / "summary.md").read_text())
    if a.record:
        BASELINE.write_text(json.dumps(results, indent=1))
        print(f"recorded {BASELINE}")
    if a.check:
        base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        req = json.loads(REQUIREMENTS.read_text()) if REQUIREMENTS.exists() else {}
        fails = check(results, base, req)
        for f in fails:
            print(f)
        print("CHECK", "FAILED" if fails else "PASSED", f"({len(fails)} issues)")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
