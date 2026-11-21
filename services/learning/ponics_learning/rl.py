"""PPO dosing policy (stable-baselines3) trained on the train suite.

Environment = one 35-day crop cycle, one action per 15 min.

Observation (10, all from sensors or the clock, same information as the MPC):
    (ec - sp)/0.3, (ph - sp)/0.5, ec and ph change since last step (x10),
    sin/cos hour, lights, age/35, last A+B dose/100, last acid-minus-base dose/40
Action (2) in [-1, 1]:
    a0 -> A+B mL = 50 * max(a0, 0)
    a1 -> acid = 15 * max(a1, 0), base = 15 * max(-a1, 0)   (never both)
Reward = -(stage cost of the MPC on the *true* EC and pH + chemical cost) / 10, and -1 extra
per step with true pH outside [5.0, 7.0]. Training uses the truth for the reward only;
the policy acts on sensor readings, like every other controller.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from ponics_bench.runner import observe
from ponics_bench.scenarios import Scenario
from ponics_contracts import CONTROL_DT_S, DAY_S, Controller, Dose, Observation, Setpoints
from ponics_control.mpc import MpcParams
from ponics_sensors import Probes
from ponics_sim.reservoir import NftSystem

from ponics_learning.data import TRAIN_SEED_MIN, check_seed

N_OBS = 10
AB_MAX, ADJ_MAX = 50.0, 15.0  # v1 used the full pump range (100, 40): 3 M steps gave a policy
# that alternated acid and base (6.8 L + 3.8 L per cycle vs 3.3 + 0.1 for MPC). Finer resolution.


class ObsBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.last = (0.0, 0.0)

    def __call__(self, o: Observation) -> np.ndarray:
        sp = o.setpoints
        dec, dph = (0.0, 0.0) if self.prev is None else (o.ec - self.prev[0], o.ph - self.prev[1])
        self.prev = (o.ec, o.ph)
        a = 2.0 * math.pi * o.hour / 24.0
        return np.array([(o.ec - sp.ec) / 0.3, (o.ph - sp.ph) / 0.5, 10 * dec, 10 * dph, math.sin(a), math.cos(a),
                         1.0 if o.lights_on else 0.0, o.day / 35.0, self.last[0] / AB_MAX, self.last[1] / ADJ_MAX],
                        dtype=np.float32)


def action_to_dose(a: np.ndarray) -> Dose:
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    return Dose(AB_MAX * max(a[0], 0.0), ADJ_MAX * max(a[1], 0.0), ADJ_MAX * max(-a[1], 0.0))


def stage_cost(ec: float, ph: float, d: Dose, sp: Setpoints, p: MpcParams = MpcParams()) -> float:
    track = p.q_ec * (ec - sp.ec) ** 2 + p.q_ph * (ph - sp.ph) ** 2
    chem = p.cost[0] * d.ab_ml + p.cost[1] * d.acid_ml + p.cost[2] * d.base_ml
    return track + chem


class PonicsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed0: int = 2000, days: int = 35):
        check_seed(seed0, "train")
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.seed0, self.days, self.episode = seed0, days, 0
        self.sp = Setpoints()
        self.build = ObsBuilder()

    def reset(self, *, seed=None, options=None):
        # The gym seed only seeds gym's own RNG. Scenarios always come from this env's
        # training-seed range: SB3 passes seeds 0..n_envs-1 here, which are eval seeds.
        super().reset(seed=seed)
        sc_seed = self.seed0 + self.episode
        check_seed(sc_seed, "train")
        sc = Scenario("train", sc_seed, "sensor", self.days)
        self.episode += 1
        self.sys = NftSystem(sc.config())
        self.probes = Probes(sc.sensor_seed)
        self.steps, self.max_steps = 0, int(self.days * DAY_S / CONTROL_DT_S)
        self.build.reset()
        return self.build(observe(self.sys, self.probes, self.sp)), {}

    def step(self, action):
        dose = self.sys.step(action_to_dose(action))
        self.build.last = (dose.ab_ml, dose.acid_ml - dose.base_ml)
        self.steps += 1
        ph, ec = self.sys.ph, self.sys.ec
        r = -stage_cost(ec, ph, dose, self.sp) / 10.0
        if ph < 5.0 or ph > 7.0:
            r -= 1.0
        r = max(r, -20.0)
        obs = self.build(observe(self.sys, self.probes, self.sp))
        truncated = self.steps >= self.max_steps
        return obs, float(r), False, truncated, {"fw": self.sys.fresh_weight_g() if truncated else None}


class PpoController(Controller):
    def __init__(self, path: Path):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(path), device="cpu")
        self.build = ObsBuilder()
        self.name = "ppo"

    def reset(self) -> None:
        self.build.reset()

    def act(self, obs: Observation) -> Dose:
        a, _ = self.model.predict(self.build(obs), deterministic=True)
        d = action_to_dose(a)
        self.build.last = (d.ab_ml, d.acid_ml - d.base_ml)
        return d


def _make(rank: int, seed0: int):
    def f():
        import torch

        torch.set_num_threads(1)
        return PonicsEnv(seed0 + 100 * rank)

    return f


def train(out: Path, timesteps: int = 6_000_000, n_envs: int = 10, seed: int = 0, seed0: int = TRAIN_SEED_MIN + 1000) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    env = VecMonitor(SubprocVecEnv([_make(i, seed0) for i in range(n_envs)]))
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2, policy_kwargs={"net_arch": [64, 64]},
                seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=timesteps)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    env.close()
    return {"timesteps": timesteps, "n_envs": n_envs, "seed": seed, "seed0": seed0}
