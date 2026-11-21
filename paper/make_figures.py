"""Figures and tables for the paper, from recorded data only.

Inputs: results/latest/results.json (ponics-eval), results/replay/replay.json (ponics-replay),
paper/data/sweep.json (paper/sweep.py), models/gp.pkl. Outputs: paper/figures/*.pdf|png and
paper/tables/*.tex. Run: uv run python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "paper" / "figures", ROOT / "paper" / "tables"
COL = {"rule": "#8a6d3b", "pid": "#c0392b", "mpc": "#2b6cb0", "gp_mpc": "#1f8a5b", "mpc_tight": "#8e44ad",
       "gp_mpc_tight": "#d4821a", "ppo": "#4a4a4a"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_buffer():
    """True per-mL pH gain of the train-suite solution vs the nominal and the GP-corrected gain."""
    from scipy.optimize import brentq

    from ponics_bench.scenarios import Scenario
    from ponics_chem import co2_saturation_mM, speciate
    from ponics_contracts.residual import features
    from ponics_control.nominal import NominalModel
    from ponics_learning.gp import ResidualGp
    from ponics_sim.reservoir import NftSystem

    s = NftSystem(Scenario("train", 0).config())
    base = s.conc
    sat = co2_saturation_mM(800.0)

    def ph_at(acid_mM, ct):
        d = dict(base)
        d["NO3"] += acid_mM
        d["C"] = ct
        return speciate(d)

    phs = np.linspace(5.0, 7.0, 21)
    beta = []
    for ph in phs:
        ct = 0.05
        for _ in range(25):  # C_T in equilibrium with 800 ppm room air at this pH
            a = brentq(lambda x: ph_at(x, ct).ph - ph, -10, 30)
            ct *= sat / max(ph_at(a, ct).co2_aq, 1e-9)
        a2 = brentq(lambda x: ph_at(x, ct).ph - (ph - 0.05), -10, 30)
        beta.append((a2 - a) / 0.05)
    beta = np.array(beta)
    g_true = 0.5 / (beta * 100.0)
    g_nom = NominalModel().B(100.0)[1, 2]
    gp = ResidualGp.load(ROOT / "models" / "gp.pkl")
    out = {}
    for age, lab in ((0.25, "day 9"), (0.75, "day 26")):
        Z = np.array([features(p, 1.8, 12.0, True, age * 35) for p in phs])
        _, b = gp.parts(Z)
        inside = gp.support(Z)
        out[lab] = (np.where(inside, g_nom - b, np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.35})
    ax[0].plot(phs, beta, color="k")
    ax[0].axhline(0.30, color=COL["mpc"], ls="--", label="nominal (jar test)")
    ax[0].set_xlabel("pH")
    ax[0].set_ylabel(r"buffer capacity $\beta$ (mM/pH)")
    ax[0].legend(frameon=False)
    ax[1].plot(phs, g_true * 1e3, color="k", label="true, fresh solution")
    ax[1].axhline(g_nom * 1e3, color=COL["mpc"], ls="--", label="nominal")
    for (lab, g), ls in zip(out.items(), ("-", ":")):
        ax[1].plot(phs, g * 1e3, color=COL["gp_mpc"], ls=ls, label=f"GP-corrected, {lab}")
    ax[1].set_xlabel("pH")
    ax[1].set_ylabel(r"$|\Delta$pH$|$ per mL acid ($\times10^{-3}$)")
    ax[1].set_yscale("log")
    ax[1].legend(frameon=False, fontsize=7)
    save(fig, "buffer")
    return {"beta_min": float(beta.min()), "beta_max": float(beta.max())}


def fig_traces():
    d = json.loads((ROOT / "results" / "replay" / "replay.json").read_text())
    days = np.arange(d["hours"]) / 24
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 3.6), sharex=True)
    for a, suite, names in ((ax[0], "train", ["mpc_tight", "mpc", "gp_mpc_tight"]),
                            (ax[1], "unseen", ["pid", "rule", "ppo", "mpc"])):
        a.axhspan(5.5, 6.5, color="0.92", zorder=0)
        for n in names:
            a.plot(days, d["runs"][suite][n]["ph"], color=COL[n], lw=0.8 if n != names[-1] else 1.2, label=n)
        for dd in d["runs"][suite][names[0]]["dump_days"]:
            a.axvline(dd + 8 / 24, color="0.5", ls="--", lw=0.6)
        a.set_ylim(4.0, 8.0)
        a.set_ylabel("true pH")
        a.legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
        a.set_title({"train": "train suite (soft water), seed 0", "unseen": "unseen suite (hard water, 28 °C), seed 0"}[suite], fontsize=8)
    ax[1].set_xlabel("day after transplant")
    save(fig, "traces")


def fig_sweep():
    rows = json.loads((ROOT / "paper" / "data" / "sweep.json").read_text())
    pens = sorted({r["r_du"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"wspace": 0.35})
    stats = {}
    for kind in ("mpc", "gp_mpc"):
        for suite, ls in (("train", "-"), ("unseen", "--")):
            for j, met in enumerate(("ph_rms", "ph_severe_h")):
                m = np.array([[r[met] for r in rows if r["controller"] == kind and r["suite"] == suite and r["r_du"] == p] for p in pens])
                mu, sd = m.mean(1), m.std(1)
                lo = np.minimum(sd, mu - (1e-3 if met == "ph_rms" else 0.0))  # error bars never below zero
                ax[j].errorbar(pens, mu, yerr=[lo, sd], color=COL[kind], ls=ls, marker="o", ms=3, capsize=2,
                               label=f"{'GP-MPC' if kind == 'gp_mpc' else 'MPC'}, {suite}")
                stats[(kind, suite, met)] = m.mean(1).tolist()
    for a, lab in zip(ax, ("pH RMS", "hours with pH < 5 or > 7")):
        a.set_xscale("log")
        a.set_xticks(pens)
        a.set_xticklabels([f"{p:g}" for p in pens])
        a.minorticks_off()
        a.set_xlabel(r"move penalty on net acid $r_{\Delta}$")
        a.set_ylabel(lab)
    ax[0].set_yscale("log")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylim(bottom=0)
    ax[0].legend(frameon=False, fontsize=7)
    save(fig, "sweep")
    return {"penalties": pens, "stats": {"|".join(k): v for k, v in stats.items()}}


def tables():
    r = json.loads((ROOT / "results" / "latest" / "results.json").read_text())
    names = ["rule", "pid", "mpc", "gp_mpc", "mpc_tight", "gp_mpc_tight", "ppo"]
    label = {"rule": "Rule", "pid": "PID", "mpc": "MPC", "gp_mpc": "GP-MPC", "mpc_tight": "MPC, $r_\\Delta{=}0.003$",
             "gp_mpc_tight": "GP-MPC, $r_\\Delta{=}0.003$", "ppo": "PPO"}
    TAB.mkdir(parents=True, exist_ok=True)
    for suite in ("train", "unseen"):
        lines = []
        for n in names:
            m = r[f"{n}|{suite}|sensor"]
            lines.append(f"{label[n]} & {100 * m['ph_in_band']:.1f} & {m['ph_rms']:.3f} $\\pm$ {m['ph_rms_sd']:.3f} & "
                         f"{m['ph_severe_h']:.1f} & {100 * m['ec_in_band']:.1f} & {m['fw_g']:.0f} $\\pm$ {m['fw_g_sd']:.0f} & "
                         f"{m['acid_l']:.2f} & {m['base_l']:.2f} & {m['root_health']:.2f} \\\\")
        (TAB / f"results_{suite}.tex").write_text("\n".join(lines) + "\n")


def numbers(info: dict) -> None:
    """LaTeX macros for every number quoted in the text, so prose cannot drift from data."""
    r = json.loads((ROOT / "results" / "latest" / "results.json").read_text())
    log = json.loads((ROOT / "models" / "train_log.json").read_text())
    gp = [e for e in log if e["model"] == "gp"][-1]
    ppo = [e for e in log if e["model"] == "ppo"][-1]
    m = {}

    def put(name, val, fmt):
        m[name] = fmt.format(val)

    key = lambda c, s: r[f"{c}|{s}|sensor"]  # noqa: E731
    for c, tag in (("rule", "Rule"), ("pid", "Pid"), ("mpc", "Mpc"), ("gp_mpc", "Gp"), ("mpc_tight", "MpcT"),
                   ("gp_mpc_tight", "GpT"), ("ppo", "Ppo")):
        for s, st in (("train", "Tr"), ("unseen", "Un")):
            x = key(c, s)
            put(f"{tag}{st}Rms", x["ph_rms"], "{:.3f}")
            put(f"{tag}{st}Band", 100 * x["ph_in_band"], "{:.1f}")
            put(f"{tag}{st}Severe", x["ph_severe_h"], "{:.1f}")
            put(f"{tag}{st}Fw", x["fw_g"], "{:.0f}")
            put(f"{tag}{st}Acid", x["acid_l"], "{:.2f}")
            put(f"{tag}{st}Base", x["base_l"], "{:.2f}")
            put(f"{tag}{st}Root", x["root_health"], "{:.2f}")
            put(f"{tag}{st}EcBand", 100 * x["ec_in_band"], "{:.1f}")
            put(f"{tag}{st}EcLow", 100 * x["ec_low_frac"], "{:.1f}")
    put("GpRtwo", gp["r2_dph_held_out"], "{:.2f}")
    put("GpRmseNom", gp["rmse_dph_nominal"], "{:.3f}")
    put("GpRmseGp", gp["rmse_dph_gp"], "{:.3f}")
    put("GpPoints", gp["points"], "{:d}")
    put("GpEpisodes", gp["episodes"], "{:d}")
    put("GpSeconds", gp["seconds"], "{:.0f}")
    put("PpoSteps", ppo["timesteps"] / 1e6, "{:.0f}")
    put("BetaMin", info["buffer"]["beta_min"], "{:.3f}")
    put("BetaMax", info["buffer"]["beta_max"], "{:.2f}")
    put("BetaRatio", info["buffer"]["beta_max"] / info["buffer"]["beta_min"], "{:.0f}")
    if "sweep" in info:
        st, pens = info["sweep"]["stats"], info["sweep"]["penalties"]
        for kind, kt in (("mpc", "Mpc"), ("gp_mpc", "Gp")):
            for s, sst in (("train", "Tr"), ("unseen", "Un")):
                for met, mt in (("ph_rms", "Rms"), ("ph_severe_h", "Sev")):
                    for i, p in enumerate(pens):
                        tag = {0.003: "A", 0.01: "B", 0.02: "C", 0.05: "D"}[p]
                        put(f"Sw{kt}{sst}{mt}{tag}", st[f"{kind}|{s}|{met}"][i], "{:.3f}" if met == "ph_rms" else "{:.0f}")
    lines = [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (TAB / "numbers.tex").write_text("% generated by paper/make_figures.py, do not edit\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    info = {"buffer": fig_buffer()}
    fig_traces()
    if (ROOT / "paper" / "data" / "sweep.json").exists():
        info["sweep"] = fig_sweep()
    tables()
    numbers(info)
    (ROOT / "paper" / "data" / "figure_numbers.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
