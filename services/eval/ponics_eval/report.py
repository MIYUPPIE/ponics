"""summary.md table and example-episode plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLS = [("ph_in_band", "pH in band", "{:.1%}"), ("ph_rms", "pH RMS", "{:.3f}"), ("ph_severe_h", "pH severe h", "{:.1f}"),
        ("ec_low_frac", "EC low", "{:.1%}"), ("ec_in_band", "EC in band", "{:.1%}"), ("fw_g", "FW g", "{:.0f}"),
        ("stock_l", "A+B L", "{:.2f}"), ("acid_l", "acid L", "{:.2f}"), ("base_l", "base L", "{:.2f}"),
        ("cost", "cost", "{:.2f}"), ("root_health", "root", "{:.2f}"), ("solve_p99_ms", "p99 ms", "{:.1f}")]


def write_summary(results: dict, path: Path, header: str = "") -> None:
    lines = [header, "", "| controller | suite | feedback | " + " | ".join(c[1] for c in COLS) + " |",
             "|" + "---|" * (3 + len(COLS))]
    for key, r in results.items():
        ctrl, suite, fb = key.split("|")
        lines.append(f"| {ctrl} | {suite} | {fb} | " + " | ".join(f.format(r[k]) for k, _, f in COLS) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_examples(logs: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for name, log in logs.items():
        a = log.arrays()
        d = a["t"] / 86400.0
        ax[0].plot(d, a["ph"], lw=0.8, label=name)
        ax[1].plot(d, a["ec"], lw=0.8, label=name)
        ax[2].plot(d, np.cumsum(a["acid_ml"] + a["base_ml"]) / 1000.0, lw=1.0, label=name)
    sp = next(iter(logs.values())).setpoints
    ax[0].axhspan(sp.ph_lo, sp.ph_hi, color="0.9", zorder=0)
    ax[0].set_ylabel("true pH")
    ax[0].set_ylim(4.0, 8.0)
    ax[1].axhspan(sp.ec - sp.ec_band, sp.ec + sp.ec_band, color="0.9", zorder=0)
    ax[1].set_ylabel("true EC25 dS/m")
    ax[2].set_ylabel("acid + base used, L")
    ax[2].set_xlabel("day")
    ax[0].legend(ncol=5, fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
