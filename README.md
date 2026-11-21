# ponics

Nutrient-solution control for a recirculating NFT lettuce reservoir. One simulator, one
scenario set and one metric, used to compare a grower's rule doser, PID, a nominal MPC,
a GP-MPC with a learned pH residual, and PPO.

**Question:** the pH buffer of a nutrient solution varies 13x across the operating range
and drifts as the crop removes phosphate and the tap water brings alkalinity. A model
from one jar test is right in structure and wrong in detail. How much does a learned
residual (GP-MPC) or a model-free policy (PPO) recover, at what data cost, and does it
survive different water and a hotter room?

Origin: angle 1 of the novelty check in `CV/apmr/novelty-check.md` (no credible paper
compares learned control of EC/pH dosing against tuned PID/MPC). The method mirrors
`furrow` (GP-residual MPC vs RL), which follows Dr. Jie Wang's GP-FBLMPC.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Charge-balance pH / Onsager EC chemistry, crop and reservoir simulator, probe models, rule, PID, offset-free MPC, seeded eval harness | **done** |
| 2 | Structured GP residual (drift + buffer-gain), GP-MPC, tuning-robustness study | **done** |
| 3 | PPO dosing policy on the same objective | **done** |
| 4 | Hardware: one or two real reservoirs, ESP32 + Atlas-class probes + peristaltic pumps | planned, `docs/HARDWARE.md` |

## Results

RESULTS_PLACEHOLDER

## Findings so far

1. **Nominal MPC needs move suppression to survive the real buffer.** Near pH 5.6 the true
   gain is 2.6x the jar-test value; the near-deadbeat MPC fell into an acid/base limit
   cycle at full pump rate. The penalty must act on net acid (acid - base); per-pump
   penalties made the optimiser dose both.
2. **PID is marginally stable** for the same reason: kicked toward pH 5 it limit-cycles.
3. **An unstructured GP residual made control worse** even with held-out R^2 0.92: its
   slope in the dose was up to 5x wrong, because in closed-loop data acid is dosed exactly
   when pH rises. A structured residual a(z) + b(z) (acid - base), logging with random holds,
   and a training-support fallback fixed it.
4. **The GP's value is safe aggressive tuning.** In distribution, GP-MPC runs stable at a
   17x smaller move penalty, where the nominal MPC kills the roots, and tracks best of all.
   On hard water and heat the soft-water GP does not transfer; the support box does not
   detect the shift because it is in the dynamics, not the inputs.
5. **Composition drift is real and uncontrollable with EC alone.** Holding EC lets Ca, Mg,
   SO4 and Na build up, phosphate runs out late in the cycle, the buffer collapses, and
   EC cannot be brought down without dumping. Ion-specific sensing is the next research angle.

Details, equations, parameters and every dead end: `docs/DESIGN.md`.

## Layout

```
contracts/ponics_contracts/   shared types (Dose, Observation, Setpoints, limits) and the residual contract
services/chem/                pH (charge balance, Davies activity), EC (Onsager), recipes, tap waters
services/sim/                 crop (LUE growth, transpiration, uptake, stress), climate, reservoir + ledger
services/sensors/             pH and EC probe models (drift, recalibration, temperature compensation)
services/control/             rule, PID, nominal model, offset-free MPC (OSQP), GP-MPC
services/learning/            residual data logging, structured GP, PPO env + training, ponics-train
services/bench/               scenarios (suites, seeds), closed-loop episode runner
services/eval/                metrics, registry, report, ponics-eval
evals/                        baseline.json (recorded run), requirements.json (hard limits)
scripts/                      gate.sh, eval.sh
docs/                         DESIGN.md, HARDWARE.md
```

Services depend only on `ponics_contracts` and on services below them
(eval -> learning -> control -> bench -> sensors, sim -> chem). Each has its own `tests/`.

## Run

```bash
uv sync
scripts/gate.sh                  # gate tests, also run by the pre-commit hook
uv run ponics-train gp           # residual GP, ~80 s (train suite, seeds >= 1000)
uv run ponics-train ppo          # PPO, ~36 min on 12 cores
scripts/eval.sh                  # trains anything missing, 140 seeded 35-day episodes, regression check
uv run ponics-eval --controllers mpc,gp_mpc --seeds 3        # quick partial run
uv run ponics-eval --record      # deliberately accept a new baseline (full run only)
uv run pre-commit install        # once per clone
```

`scripts/*.sh` clear `PYTHONPATH` so a sourced ROS 2 install cannot inject pytest plugins.
torch comes from the PyTorch CPU index (see `pyproject.toml`).

## Evaluation contract

* Scenario = (suite, seed, feedback). It fixes tap water, room climate, cultivar jitter and
  all probe noise. Seeds 0..99 are evaluation seeds; learning code refuses them.
* `train`: soft tap water (alkalinity 2 mM), 22/18 C, dumps on days 14 and 28.
  `unseen`: hard tap water (alkalinity 3.8 mM, Ca 2 mM), 28/22 C at low humidity, weekly
  dumps. No learned controller trains on it.
* Setpoints EC 1.8 dS/m, pH 5.8; band 5.5 to 6.5 and EC +/- 0.2.
* Metrics on the true state: pH in band, pH RMS, severe hours (pH < 5 or > 7), EC low,
  fresh weight, chemicals, a common cost (the MPC stage cost), root health, decision time.
* `--check` fails on >10% + 0.005 regression of any metric vs `evals/baseline.json`, any
  break of `evals/requirements.json`, or p99 decision time over 200 ms.

## References

* J. Wang et al., GP-based learning MPC for mobile-robot path following (GP-FBLMPC). Robot
  Autonomy Lab, University of Manitoba.
* L. Hewing, J. Kabzan, M. N. Zeilinger, "Cautious model predictive control using Gaussian
  process regression", IEEE TCST, 2020.
* C. Sonneveld, W. Voogt, *Plant Nutrition of Greenhouse Crops*, Springer, 2009.
* W. Stumm, J. J. Morgan, *Aquatic Chemistry*, 3rd ed., 1996 (charge balance, Davies).
* E. J. van Henten, "Greenhouse climate management: an optimal control approach", 1994 (lettuce growth).
* Novelty and related work: `CV/apmr/novelty-check.md`.
