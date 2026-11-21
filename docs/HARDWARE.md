# Phase 4: hardware validation plan

Goal: run the best simulated controller and the PID baseline on one real reservoir and
check that the simulated ranking holds. A single 35-day lettuce cycle per controller is
the minimum; two reservoirs side by side remove the season as a confound.

## Rig (per reservoir)

| part | role | interface |
|---|---|---|
| 60 to 100 L opaque tank, air pump + stone | reservoir, CO2 exchange | - |
| NFT channels, 72 sites or fewer | crop | - |
| float valve on tap line + inline flow meter | top-up, water ledger | pulse input |
| isolated pH probe + EC (K 1.0) probe, Atlas Scientific EZO class | the two sensors the controllers use | I2C, isolated carrier |
| DS18B20 in the tank | water temperature for EC compensation | 1-Wire |
| 4 peristaltic dosing pumps (A, B, acid, base), ~100 mL/min | actuators | MOSFET/driver, timed |
| ESP32 | reads sensors every 15 min, runs the pumps | Wi-Fi / serial to the controller host |
| scale (optional) | weekly plant fresh weight sampling | - |

The controller itself runs on a host (laptop or Pi) using the same `Controller.act`
interface; only `observe()` and `step()` are replaced by the ESP32 link. Pump volumes are
calibrated per pump (time -> mL) before each cycle.

## Protocol

1. Calibrate both probes (pH 4/7, EC 1.413); log the calibration offsets.
2. Jar-test the fresh solution to get the nominal `EC_FULL` and `BETA` for this tap water.
3. Cycle 1: PID and MPC. Log readings, doses and top-up every step. Weekly hand
   readings with a second, freshly calibrated meter give the "truth".
4. Fit the GP residual on cycle-1 logs (`ponics_learning`, same contract).
5. Cycle 2: GP-MPC and PID. Same metrics as `ponics_eval.metrics`, computed on the hand
   readings and the logged readings.
6. Harvest: fresh weight per plant.

## Safety on the rig

Hard limits in firmware, independent of the controller: max dose per step (as in
`PumpLimits`), and a pH stop band. Below pH 4.8 or above 7.2 the ESP32 refuses further
acid or base respectively and raises an alert.
