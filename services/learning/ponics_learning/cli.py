"""ponics-train: fit the learned models. Train suite, seeds >= 1000 only.

  ponics-train gp            residual GP -> models/gp.pkl     (~80 s)
  ponics-train ppo           PPO policy  -> models/ppo.zip    (~70 min, 10 envs, 6 M steps)
  ponics-train all
  ponics-train gp --episodes 2 --points 300     quick fit for a smoke test

Each run appends what it did and its held-out score to models/train_log.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ponics_learning.data import TRAIN_SEED_MIN, collect
from ponics_learning.gp import ResidualGp

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
GP_PATH = MODELS / "gp.pkl"
PPO_PATH = MODELS / "ppo.zip"


def _log(entry: dict) -> None:
    MODELS.mkdir(exist_ok=True)
    p = MODELS / "train_log.json"
    log = json.loads(p.read_text()) if p.exists() else []
    log.append(entry)
    p.write_text(json.dumps(log, indent=1))


def subsample(n_dose: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Indices: every step with > 2 mL net acid or base (they carry the buffer-gain
    information) up to m/2, the rest uniformly at random. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    dosed = np.flatnonzero(np.abs(n_dose) > 2.0)
    rest = np.setdiff1d(np.arange(len(n_dose)), dosed)
    if len(dosed) > m // 2:
        dosed = rng.choice(dosed, m // 2, replace=False)
    take = np.concatenate([dosed, rng.choice(rest, min(m - len(dosed), len(rest)), replace=False)])
    take.sort()
    return take


def train_gp(episodes: int, points: int, days: int = 35) -> dict:
    t0 = time.time()
    seeds = [TRAIN_SEED_MIN + i for i in range(episodes + 1)]  # last one is held out
    with ProcessPoolExecutor(max_workers=min(len(seeds), os.cpu_count() or 1)) as ex:
        data = list(ex.map(collect, seeds, [days] * len(seeds)))
    Z = np.vstack([d[0] for d in data[:-1]])
    N = np.concatenate([d[1] for d in data[:-1]])
    Y = np.concatenate([d[2] for d in data[:-1]])
    idx = subsample(N, points, seed=0)
    gp = ResidualGp().fit(Z[idx], N[idx], Y[idx])
    Zh, Nh, Yh = data[-1]
    pred = gp.predict(Zh, Nh)
    r2 = 1 - np.mean((Yh - pred) ** 2) / np.var(Yh)
    gp.meta |= {"seeds": seeds[:-1], "held_out": seeds[-1], "points": len(idx), "r2_held_out": float(r2)}
    gp.save(GP_PATH)
    entry = {"model": "gp", "seconds": round(time.time() - t0, 1), "episodes": episodes, "points": len(idx),
             "held_out_seed": seeds[-1], "r2_dph_held_out": round(float(r2), 3),
             "rmse_dph_nominal": round(float(np.sqrt(np.mean(Yh**2))), 4),
             "rmse_dph_gp": round(float(np.sqrt(np.mean((Yh - pred) ** 2))), 4),
             "hyper": gp.meta["hyper"]}
    _log(entry)
    return entry


def train_ppo(timesteps: int, n_envs: int) -> dict:
    from ponics_learning.rl import train

    t0 = time.time()
    meta = train(PPO_PATH, timesteps=timesteps, n_envs=n_envs)
    entry = {"model": "ppo", "seconds": round(time.time() - t0, 1)} | meta
    _log(entry)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ponics-train")
    ap.add_argument("what", choices=["gp", "ppo", "all"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--points", type=int, default=1000)
    ap.add_argument("--timesteps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=min(10, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    if a.what in ("gp", "all"):
        print(json.dumps(train_gp(a.episodes, a.points)))
    if a.what in ("ppo", "all"):
        print(json.dumps(train_ppo(a.timesteps, a.envs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
