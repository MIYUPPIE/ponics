# ponics design notes

What is modelled, why, the numbers behind each parameter, and what went wrong on the way.
Read with `services/*/README`-level docstrings: every module states its own equations.

## 1. The question

A recirculating NFT lettuce system has one reservoir. Its pH and EC are controlled by
dosing two nutrient stocks (A, B), acid and base. Growers run threshold dosers or PI
loops tuned on fresh solution. The plant is easy to describe and hard to model: pH
depends on a buffer that changes with pH itself, with dissolved CO2, with tap-water
alkalinity and with phosphate the crop removes. EC drifts as ions the crop does not take
(Ca, Mg, SO4, Na) accumulate.

The project compares, on one simulator and one metric:

| controller | model | data |
|---|---|---|
| rule | none, fixed shots | none |
| pid | gains from the nominal model | none |
| mpc | nominal linear model + offset-free drift estimate | one jar test |
| gp_mpc | nominal + structured GP residual on pH | 6 crop cycles of logged data, train suite |
| ppo | none (model-free) | 3 M simulated steps, train suite |
| mpc_tight, gp_mpc_tight | as above, move penalty 0.003 instead of 0.05 | as above |

and asks how much each recovers when the water and the room change (unseen suite).

## 2. Truth simulator

### 2.1 Chemistry (`ponics_chem`)

* Components: strong ions K, Ca, Mg, Na, NO3, SO4, Cl; weak totals P (phosphate) and C
  (inorganic carbon). Acid (HNO3) and base (KOH) add only their strong ion; H+/OH- appear
  through the charge balance.
* pH from electroneutrality, solved for log10[H+] with Brent's method. Conditional
  constants from Davies activity coefficients at the iterated ionic strength.
  Constants at 25 C: pKw 13.995; phosphoric 2.148 / 7.198 / 12.35; carbonic 6.352 / 10.329;
  Henry CO2 10^-1.468 mol/(L atm).
* EC from limiting equivalent conductances (CRC) with a Debye-Hueckel-Onsager correction,
  B1 = 0.2289, B2 = 60.32 split per ion, scaled by |z|.
* Validated in gate tests: pure water 7.00; 1 mM HNO3 3.015; 0.01 M KCl 1.412 mS/cm
  (reference 1.413); charge balance closes to 1e-6.
* Not modelled: ion pairing (MgSO4, CaSO4; EC of sulfate-rich solution is a few percent
  high), temperature dependence of the constants, precipitation (Ca phosphates above pH ~7).

Buffer capacity of the train-suite solution at equilibrium with 800 ppm room CO2
(mM of strong acid per pH unit):

| pH | 5.0 | 5.5 | 5.8 | 6.2 | 6.5 | 7.0 |
|---|---|---|---|---|---|---|
| beta | 0.069 | 0.114 | 0.199 | 0.413 | 0.644 | 0.921 |

13x across the operating range. Fresh, un-degassed solution is near 0.95 at 5.8 because
tap bicarbonate is still present. This is the nonlinearity the nominal model misses.

### 2.2 Recipe and water (`ponics_chem.recipes`)

Full strength (1 L A + 1 L B per 100 L), mM: NO3 12, H2PO4 1.5, SO4 1.75, K 7, Ca 3.5,
Mg 1.5 (leafy-greens range, Sonneveld & Voogt 2009). A = Ca(NO3)2 + KNO3;
B = KNO3 + KH2PO4 + MgSO4 + K2SO4. Acid and base are 0.5 M.

| tap | Ca | Mg | Na | K | Cl | SO4 | NO3 | pH | alkalinity |
|---|---|---|---|---|---|---|---|---|---|
| soft (train) | 1.0 | 0.3 | 1.0 | 0.05 | 1.0 | 0.3 | 0.1 | 7.6 | ~2.0 mM |
| hard (unseen) | 2.0 | 0.8 | 1.5 | 0.1 | 1.5 | 0.8 | 0.3 | 7.8 | ~3.8 mM |

The tank starts at 100 L mixed the way a grower does it: tap water, A+B to EC 1.8, acid
to pH 5.8, total volume 100 L.

### 2.3 Crop (`ponics_sim.crop`)

72 butterhead plants at 20 plants/m2, 35 days from transplant, LED 250 umol/m2/s for 16 h.
Light-use-efficiency growth (LUE 0.55 g DW/mol at 400 ppm, CO2 factor
1 + 0.25 log2(ppm/400)), SLA 0.035 m2/g, k 0.8, maintenance 0.012/day Q10 2.
Transpiration: radiation + VPD terms, 15% of the VPD term at night.
Nutrient uptake is demand-driven (tissue content x growth, Michaelis-Menten in
concentration) and runs on a 12 h smoothed growth rate, so it continues at night. Na and
Cl enter by mass flow with 85% / 70% exclusion.

Tissue contents (mmol/g DW): N 3.2, P 0.19, K 1.8, Ca 0.30, Mg 0.14, S 0.09. Against the
recipe ratios this means Ca, Mg and SO4 are taken up far slower than supplied, so they
accumulate when EC is held constant. That is the measured behaviour of recirculating NFT
(composition drifts at constant EC; see `CV/apmr/novelty-check.md`, PMC11214494).

Stress: pH factor 1 on [5.5, 6.5], -0.3 per pH unit for the first unit outside, then
steeper; salinity factor from EC 2.5; nutrient supply factor 1 above 40% of full strength.
Root health falls 2%/h, irreversibly, while pH < 4.5, pH > 7.8 or EC > 4.

Result under good control: about 150 g fresh weight at day 35 in the train room, 80 to
90 g in the 28 C unseen room.

### 2.4 Reservoir

100 L, float valve tops up with tap water every 5 min physics step, aerated
(k_La 2/h CO2 exchange with room air), root respiration adds CO2. Scheduled half-tank
dumps (train: days 14 and 28; unseen: weekly) are applied to every controller alike.
Every mmol in and out is booked in a ledger; mass balance is a gate test.

### 2.5 Sensors (`ponics_sensors`)

pH: offset random walk 0.03 pH/sqrt(day), weekly recalibration, slope error, noise 0.015.
EC: per-seed gain error 2%, temperature-compensation mismatch (probe 2.0 %/C, solution
1.9 %/C), 1% noise. All metrics are on the true values, so probe error counts against
every controller equally.

## 3. Controllers

### 3.1 Nominal model (`ponics_control.nominal`)

    EC+ = EC + (EC_full / (10 V)) u_ab + d_ec,     EC_full = 1.80 (jar slope)
    pH+ = pH + (0.5 / (beta V)) (u_base - u_acid) + d_ph,   beta = 0.30 mM/pH

d is an offset-free drift estimate (gain 0.1, clipped to 0.01 dS/m and 0.03 pH per step).
beta 0.30 is a jar test after a few hours of aeration. The truth ranges from 0.07 to 0.9.

### 3.2 MPC

Horizon 16 x 15 min, OSQP, box limits on the pumps.

**Finding 1, robustness.** The first MPC was near-deadbeat. Near pH 5.6 the true gain is
2.6x the nominal one, and the loop fell into an acid/base limit cycle at full pump rate
that turned the tank into KNO3 (EC 12). Fix: move suppression. Tuned on a linear plant
with 0.5x to 4x gain error (`test_mpc_robust_to_gain_error`); without it, 3x fails
(`test_unpenalised_mpc_is_not_robust`).

**Finding 2, the move penalty must act on net acid.** With separate penalties on acid
and base, cancelling acid with base was cheaper than cutting acid quickly, and the
optimiser dosed both. The penalty is now on (ab, acid - base).

### 3.3 PID

PI on EC (A+B only) and split-range PI on pH, gains from the nominal model, 0.05 pH dead
band, conditional-integration anti-windup.

**Finding 3.** PID is only marginally stable. Kicked into the weak-buffer region near
pH 5 (true gain ~4x nominal) it locks into the same acid/base limit cycle. This showed up
while logging GP data with a dithered PID, which is why the logger is the MPC.

### 3.4 Residual model and GP-MPC

**v1 (rejected).** One unstructured GP per output on (pH, EC, clock, age, ab, acid, base).
Held-out one-step R^2: pH 0.92, EC 0.12. Closed loop it was worse than the nominal MPC.
Its slopes in the dose were up to 5x wrong (at pH 5.8 it put the acid gain at 0.125 pH/mL
against ~0.025 true), because in closed-loop data acid is dosed exactly when pH is rising.
It also predicted a constant +0.039 dS/m per step EC drift that does not exist.

**v2/v3 (current).** pH only, structured:

    dpH_res = a(z) + b(z) (acid - base),   z = (pH, EC, sin h, cos h, lights, age)

as one exact GP with kernel k_a(z,z') + n n' k_b(z,z') (`ponics_learning.gp`, numpy +
scipy, marginal-likelihood fit). b is the gain correction per mL, a the drift. The MPC
stays a QP with no linearisation in u. EC uses the nominal model: mixing is linear and
the EC residual is sensor noise.

**Finding 4, identifiability.** v2 learned a +1.3 pH/step "drift" at pH 4.6 and drove the
tank further down, because in the logged data low pH was always followed by base. Two
fixes: (a) the logger now holds (doses nothing) on 10% of steps, so drift is observed
alone at every pH; (b) `support()`: outside the 1st to 99th percentile box of training
pH, EC and age, GP-MPC drops the residual and uses the nominal model. The corrected gain
is also clamped to 0.25x to 8x nominal with the nominal sign.

The marginal-likelihood gradient is analytic (checked against finite differences in
`test_marginal_likelihood_gradient_is_exact`); the fit went from ~10 min to ~80 s.

**Finding 5, what the GP buys.** At the default move penalty (0.05) the nominal MPC is
already robust and GP-MPC only trims chemical waste. The GP's value is that it makes
aggressive tuning safe. Sweep over the net-acid move penalty, 3 seeds per cell, pH RMS
and severe hours (pH < 5 or > 7):

| penalty | MPC train | GP-MPC train | MPC unseen | GP-MPC unseen |
|---|---|---|---|---|
| 0.05 | 0.074 / 0 h | 0.074 / 0 h | 0.145 / 2.2 h | 0.147 / 2.2 h |
| 0.02 | 0.069 / 0 h | 0.069 / 0 h | 0.176 / 4.4 h | 0.159 / 2.8 h |
| 0.01 | 0.067 / 0 h | 0.066 / 0 h | 0.639 / 147 h | 0.435 / 69 h |
| 0.003 | 0.624 / 160 h, roots dead | **0.063 / 0 h** | 1.114 / 359 h | 0.935 / 268 h |

In distribution, the learned gain lets the controller run 17x tighter than the nominal
model survives, and it gives the best tracking of any controller. Out of distribution
(hard water, heat) the soft-water GP does not transfer: it degrades more gracefully than
the nominal model but aggressive tuning is unsafe for both. The support box did not
catch this, because pH, EC and age stay in range; the shift is in the dynamics. Using
the GP variance to back off the tuning online is the next step. The eval grid carries
both tunings (`mpc_tight`, `gp_mpc_tight`) so this stays measured.

Fitted length scales were physically sensible: drift a depends on pH and crop age
(phosphate depletion), much less on clock; gain b on pH, EC and age.

### 3.5 PPO

stable-baselines3 PPO, MLP 64x64, 10 parallel 35-day episodes on train seeds >= 2000,
3 M steps (~36 min on 12 cores).
Observation is what the MPC sees (readings, trends, clock, age, last action). Two
actions: A+B, and one signed acid/base axis, so it can never dose both. Reward is minus
the MPC stage cost on the true state, minus 1 per step with pH outside [5, 7].

## 4. Evaluation contract

* Scenario = (suite, seed, feedback) fixes tap water, room, cultivar jitter
  (+/-10% LUE, SLA, tissue content), and all probe noise.
* Seeds 0..99 are evaluation seeds. Learning code refuses them and refuses the unseen
  suite (`check_seed`).
* Metrics on the true pH and EC: time in band, RMS, severe hours (pH < 5 or > 7), EC-low
  share (the controllable EC failure), fresh weight, chemicals, a common cost (the MPC
  stage cost), root health, solve time.
* `scripts/eval.sh` = `ponics-eval --check`: fails on >10% + 0.005 regression of any
  metric vs `evals/baseline.json`, any break of `evals/requirements.json`, or p99
  decision time over 200 ms.

## 5. Known limits

* One reservoir, one crop, one recipe. No ion-specific sensing; EC is the only nutrient
  signal, which is why Ca/Mg/SO4 build-up cannot be controlled (angle 3 of the novelty
  check).
* No precipitation chemistry, so pH > 7 does not cost phosphate or iron.
* Crop model is calibrated to typical ranges, not to a measured cultivar.
* The GP variance is not used yet. Chance constraints are the next step.
* Hardware validation (phase 4) is planned, not done: see `docs/HARDWARE.md`.
