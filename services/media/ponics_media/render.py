"""Render post-ready MP4s from a replay file (ponics-replay output).

Spec (one per output, fixed):
  wide    1920x1080, 30 fps, H.264 yuv420p, +faststart, silent
  square  1080x1080, same codec settings
Frame k is a pure function of k: matplotlib draws it from the recorded data, raw RGB goes
to FFmpeg over a pipe. No wall clock, no randomness, so re-rendering gives the same video.

Timeline of a video: TITLE_S of title card, then the 35-day cycle at one recorded hour
per frame (840 frames = 28 s), then END_S of the end card holding the final state.

  ponics-video --all                       every controller + the overview, both aspects
  ponics-video --controller gp_mpc_tight   one controller
  ponics-video --overview --aspect square
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FPS = 30
TITLE_S, END_S = 3.0, 4.0
SIZES = {"wide": (1920, 1080), "square": (1080, 1080)}
DPI = 100

BG, PANEL, INK, MUTED, RULE = "#0f1513", "#161e1b", "#e6eee9", "#8fa09a", "#2a3531"
BAND = "#1d3a2d"
OK, WARN, BAD = "#5cc08d", "#e0a13a", "#f07067"
CCOL = {"rule": "#c9a46a", "pid": "#f07067", "mpc": "#6aa8ec", "gp_mpc": "#5cc08d",
        "mpc_tight": "#c08ae0", "gp_mpc_tight": "#f0a64a", "ppo": "#b5b5b5"}
DESC = {"rule": "threshold doser, fixed shots", "pid": "PI loops tuned from one jar test",
        "mpc": "linear MPC, move penalty 0.05", "gp_mpc": "MPC + learned pH residual (GP)",
        "mpc_tight": "linear MPC, aggressive tuning", "gp_mpc_tight": "GP-MPC, aggressive tuning",
        "ppo": "model-free PPO policy"}
SUITE_TXT = {"train": "soft tap water, 22 °C room", "unseen": "hard tap water, 28 °C room (never seen in training)"}
STOPS = [(4.0, (214, 48, 39)), (5.0, (244, 128, 36)), (5.5, (246, 196, 52)), (6.0, (182, 206, 62)),
         (6.5, (86, 176, 82)), (7.0, (38, 150, 138)), (8.0, (52, 96, 184))]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": RULE})


def ph_color(p: float) -> tuple:
    if p <= STOPS[0][0]:
        c = STOPS[0][1]
    elif p >= STOPS[-1][0]:
        c = STOPS[-1][1]
    else:
        for (a, ca), (b, cb) in zip(STOPS, STOPS[1:]):
            if p <= b:
                f = (p - a) / (b - a)
                c = tuple(x + f * (y - x) for x, y in zip(ca, cb))
                break
    return tuple(v / 255 for v in c)


def status(p: float) -> tuple[str, str]:
    if p < 5.0 or p > 7.0:
        return BAD, "pH SEVERE"
    if p < 5.5 or p > 6.5:
        return WARN, "pH out of band"
    return OK, "pH in band"


def frame_hour(k: int, hours: int) -> int | None:
    """Recorded hour shown at frame k; None during the title card."""
    t0 = int(TITLE_S * FPS)
    if k < t0:
        return None
    return min(k - t0, hours - 1)


def n_frames(hours: int) -> int:
    return int(TITLE_S * FPS) + hours + int(END_S * FPS)


class Tank:
    """Reservoir glyph: tank tinted by true pH, readouts, pumps, lettuce head."""

    def __init__(self, ax, name: str, big: bool):
        self.ax, self.name = ax, name
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        fs = 1.0 if big else 0.72
        self.fs = fs
        ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=PANEL, ec=RULE, lw=1.2))
        ax.text(0.8, 9.0, name, color=CCOL.get(name, INK), fontsize=22 * fs, weight="bold", family="DejaVu Sans Mono", va="center")
        ax.add_patch(Rectangle((0.9, 2.2), 3.6, 5.2, fc="none", ec=MUTED, lw=2))
        self.water = Rectangle((1.0, 2.3), 3.4, 4.5, fc=ph_color(5.8), ec="none")
        ax.add_patch(self.water)
        ax.add_patch(Rectangle((3.7, 4.4), 0.25, 3.6, fc=MUTED, ec="none"))  # probe
        self.ph = ax.text(5.0, 7.0, "", fontsize=34 * fs, weight="bold", family="DejaVu Sans Mono", va="center")
        ax.text(5.0, 7.95, "true pH", fontsize=11 * fs, color=MUTED, va="center")
        self.probe = ax.text(5.0, 6.0, "", fontsize=12 * fs, color=MUTED, family="DejaVu Sans Mono", va="center")
        self.ec = ax.text(5.0, 5.3, "", fontsize=12 * fs, color=MUTED, family="DejaVu Sans Mono", va="center")
        self.pill = ax.text(5.0, 4.35, "", fontsize=11 * fs, weight="bold", color=BG, va="center",
                            bbox=dict(boxstyle="round,pad=0.35", fc=OK, ec="none"))
        self.pumps = [ax.text(1.0 + 2.55 * i, 1.25, lab, fontsize=10.5 * fs, family="DejaVu Sans Mono", va="center",
                              bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=RULE)) for i, lab in enumerate(("A+B", "acid", "base"))]
        self.head = Circle((8.4, 3.0), 0.3, fc="#5cc08d", ec="none")
        ax.add_patch(self.head)
        self.fw = ax.text(5.0, 3.45, "", fontsize=11 * fs, color=MUTED, family="DejaVu Sans Mono", va="center")

    def update(self, r: dict, h: int) -> None:
        p, e = r["ph"][h], r["ec"][h]
        self.water.set_facecolor(ph_color(p))
        self.ph.set_text(f"{p:.2f}")
        self.probe.set_text(f"probe {r['ph_meas'][h]:.2f}")
        arrow = " ▲" if e > 2.0 else (" ▼" if e < 1.6 else "")
        self.ec.set_text(f"EC {e:.2f} dS/m{arrow}")
        col, txt = status(p)
        self.pill.set_text(txt)
        self.pill.get_bbox_patch().set_facecolor(col)
        for t, key, lab in zip(self.pumps, ("ab", "acid", "base"), ("A+B", "acid", "base")):
            v = r[key][h]
            on = v > 0.5
            t.set_text(f"{lab} {v:.0f}mL" if on else lab)
            t.get_bbox_patch().set_edgecolor(CCOL.get(self.name, INK) if on else RULE)
            t.set_color(INK if on else MUTED)
        fw = r["fw"][h]
        self.head.set_radius(min(1.1, 0.2 + 0.07 * np.sqrt(fw)))
        self.head.set_facecolor(WARN if r["stress"][h] < 0.6 else "#5cc08d")
        self.fw.set_text(f"{fw:.0f} g/plant")
        self.fw.set_color(INK)


def trace_axes(ax, key: str, data: dict, suite: str, names: list[str], focus: str | None, sp: dict, fs: float = 1.0):
    lo, hi, band, ticks = ((4.0, 8.0, (sp["ph_lo"], sp["ph_hi"]), [4, 5, 6, 7, 8]) if key == "ph"
                           else (1.0, 3.5, (sp["ec"] - sp["ec_band"], sp["ec"] + sp["ec_band"]), [1.0, 1.8, 2.5, 3.5]))
    ax.set_facecolor(PANEL)
    ax.axhspan(*band, color=BAND, zorder=0)
    days = np.arange(data["hours"]) / 24.0
    for d in data["runs"][suite][names[0]]["dump_days"]:
        ax.axvline(d + 8 / 24, color=MUTED, ls="--", lw=0.8, zorder=1)
    lines = {}
    for n in names:
        on = focus is None or n == focus
        (ln,) = ax.plot([], [], color=CCOL.get(n, INK), lw=2.2 if on else 1.0, alpha=1.0 if on else 0.3, zorder=3 if on else 2)
        lines[n] = ln
    ax.set_xlim(0, 35)
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.set_xticks(range(0, 36, 7))
    ax.tick_params(labelsize=10 * fs)
    ax.grid(axis="y", color=RULE, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("true pH" if key == "ph" else "EC dS/m", fontsize=11 * fs)
    cursor = ax.axvline(0, color=INK, lw=1.0, zorder=4)

    def update(h: int):
        for n, ln in lines.items():
            ln.set_data(days[: h + 1], data["runs"][suite][n][key][: h + 1])
        cursor.set_xdata([days[h], days[h]])

    return update


def _encoder(path: Path, w: int, h: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _emit(fig, enc) -> None:
    fig.canvas.draw()
    enc.stdin.write(fig.canvas.buffer_rgba().tobytes())


def _title(fig, lines: list[tuple[str, float, str, str]]):
    """lines: (text, y, size, color) in figure coordinates; returns the artists."""
    return [fig.text(0.5, y, t, ha="center", va="center", fontsize=s, color=c, weight="bold" if s > 30 else "normal")
            for t, y, s, c in lines]


def render_controller(data: dict, name: str, out: Path, aspect: str = "wide", max_frames: int | None = None) -> Path:
    W, H = SIZES[aspect]
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    suites = ("train", "unseen")
    sp, hours = data["setpoints"], data["hours"]
    tanks, ups, chrome = {}, [], []
    if aspect == "wide":
        for i, s in enumerate(suites):
            y0 = 0.53 - 0.47 * i
            chrome.append(fig.text(0.02, y0 + 0.40, SUITE_TXT[s], fontsize=15, color=MUTED))
            tanks[s] = Tank(fig.add_axes([0.01, y0, 0.25, 0.38]), name, big=True)
            ups.append(trace_axes(fig.add_axes([0.31, y0 + 0.17, 0.66, 0.20]), "ph", data, s, [name], None, sp))
            ups.append(trace_axes(fig.add_axes([0.31, y0 + 0.02, 0.66, 0.12]), "ec", data, s, [name], None, sp))
    else:
        for i, s in enumerate(suites):
            y0 = 0.52 - 0.47 * i
            chrome.append(fig.text(0.03, y0 + 0.40, SUITE_TXT[s], fontsize=13, color=MUTED))
            tanks[s] = Tank(fig.add_axes([0.01, y0 + 0.02, 0.40, 0.36]), name, big=False)
            ups.append(trace_axes(fig.add_axes([0.49, y0 + 0.20, 0.48, 0.17]), "ph", data, s, [name], None, sp, fs=0.85))
            ups.append(trace_axes(fig.add_axes([0.49, y0 + 0.04, 0.48, 0.11]), "ec", data, s, [name], None, sp, fs=0.85))
    head = fig.text(0.98, 0.965, "", ha="right", fontsize=15, color=INK, family="DejaVu Sans Mono")
    chrome += [head, fig.text(0.02 if aspect == "wide" else 0.03, 0.965, f"ponics · {name} · {DESC[name]}", fontsize=15, color=INK)]
    cover = fig.add_axes([0, 0, 1, 1])
    cover.set_facecolor(BG)
    cover.axis("off")
    cover.add_patch(Rectangle((0, 0), 1, 1, transform=cover.transAxes, fc=BG))
    m = {s: data["runs"][s][name]["metrics"] for s in suites}
    title = _title(fig, [("NFT lettuce · 100 L reservoir · 35 days", 0.66, 20, MUTED),
                         (name, 0.53, 54, CCOL.get(name, INK)), (DESC[name], 0.42, 22, INK),
                         ("same crop, same probes, same tap water as every other controller", 0.32, 16, MUTED)])
    end = _title(fig, [(f"{name} · end of cycle", 0.74, 30, CCOL.get(name, INK))] + [
        (f"{SUITE_TXT[s]}:  pH in band {m[s]['ph_in_band']:.1%} · severe {m[s]['ph_severe_h']:.0f} h · "
         f"{m[s]['fw_g']:.0f} g/plant · acid+base {m[s]['acid_l'] + m[s]['base_l']:.1f} L", 0.60 - 0.09 * i, 15 if aspect == "wide" else 11, INK)
        for i, s in enumerate(suites)] + [("simulation · github.com/MIYUPPIE/ponics", 0.30, 14, MUTED)])
    enc = _encoder(out, W, H)
    total = n_frames(hours) if max_frames is None else max_frames
    for k in range(total):
        h = frame_hour(k, hours)
        in_end = k >= int(TITLE_S * FPS) + hours
        cover.set_visible(h is None or in_end)
        for t in chrome:
            t.set_visible(not (h is None or in_end))
        for t in title:
            t.set_visible(h is None)
        for t in end:
            t.set_visible(in_end)
        if h is not None:
            for s in suites:
                tanks[s].update(data["runs"][s][name], h)
            for u in ups:
                u(h)
            head.set_text(f"day {h // 24:2d}  {h % 24:02d}:00  {'LEDs on ' if 6 <= h % 24 < 22 else 'LEDs off'}")
        _emit(fig, enc)
    enc.stdin.close()
    enc.wait()
    plt.close(fig)
    return out


def render_overview(data: dict, out: Path, aspect: str = "wide", max_frames: int | None = None) -> Path:
    """Both suites back to back: soft water cycle, then hard water cycle, all controllers."""
    W, H = SIZES[aspect]
    names, sp, hours = data["controllers"], data["setpoints"], data["hours"]
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    n = len(names)
    tanks, ups = {}, {}
    if aspect == "wide":
        w = 0.96 / n
        for i, nm in enumerate(names):
            tanks[nm] = Tank(fig.add_axes([0.02 + i * w, 0.53, w - 0.006, 0.36]), nm, big=False)
        ph_ax, ec_ax = [0.05, 0.22, 0.92, 0.27], [0.05, 0.05, 0.92, 0.13]
    else:
        cols = 4
        for i, nm in enumerate(names):
            r_, c_ = divmod(i, cols)
            tanks[nm] = Tank(fig.add_axes([0.01 + c_ * 0.245, 0.62 - r_ * 0.25, 0.24, 0.24]), nm, big=False)
            for t in (tanks[nm].ph,):
                t.set_fontsize(22)
        ph_ax, ec_ax = [0.08, 0.16, 0.89, 0.18], [0.08, 0.04, 0.89, 0.09]
    for s in ("train", "unseen"):
        pa, ea = fig.add_axes(ph_ax), fig.add_axes(ec_ax)
        ups[s] = (pa, ea, trace_axes(pa, "ph", data, s, names, None, sp, fs=0.9 if aspect == "wide" else 0.75),
                  trace_axes(ea, "ec", data, s, names, None, sp, fs=0.9 if aspect == "wide" else 0.75))
    suite_lbl = fig.text(0.02, 0.965, "", fontsize=17 if aspect == "wide" else 13, color=INK)
    clock = fig.text(0.98, 0.965, "", ha="right", fontsize=15 if aspect == "wide" else 12, color=INK, family="DejaVu Sans Mono")
    cover = fig.add_axes([0, 0, 1, 1])
    cover.axis("off")
    cover.add_patch(Rectangle((0, 0), 1, 1, transform=cover.transAxes, fc=BG))
    tsz = 1.0 if aspect == "wide" else 0.75
    title = _title(fig, [("Same crop. Same probes. Seven controllers.", 0.60, 44 * tsz, INK),
                         ("pH and EC control of a recirculating NFT lettuce reservoir, simulated for 35 days", 0.48, 18 * tsz, MUTED),
                         ("rule · PID · MPC · GP-MPC · aggressive MPC / GP-MPC · PPO", 0.38, 18 * tsz, MUTED)])
    mid = _title(fig, [("Now: hard tap water and a 28 °C room.", 0.58, 36 * tsz, INK),
                       ("None of the learned controllers saw this in training.", 0.46, 20 * tsz, MUTED)])
    res = {s: {nm: data["runs"][s][nm]["metrics"] for nm in names} for s in ("train", "unseen")}
    best_train = min(names, key=lambda nm: res["train"][nm]["ph_rms"])
    best_unseen = min(names, key=lambda nm: res["unseen"][nm]["ph_rms"])
    end = _title(fig, [("What the simulation says (seed 0)", 0.80, 30 * tsz, INK),
                       (f"Soft water: best pH tracking {best_train} (RMS {res['train'][best_train]['ph_rms']:.3f}).", 0.67, 17 * tsz, INK),
                       (f"Aggressive tuning without the learned model: {res['train']['mpc_tight']['ph_severe_h']:.0f} h of severe pH, roots lost."
                        if "mpc_tight" in names else "", 0.59, 17 * tsz, INK),
                       (f"Hard water: best {best_unseen} (RMS {res['unseen'][best_unseen]['ph_rms']:.3f}); the learned model did not transfer.", 0.51, 17 * tsz, INK),
                       ("A learned residual buys a bigger stable region, not robustness to new water.", 0.40, 19 * tsz, CCOL["gp_mpc_tight"]),
                       ("github.com/MIYUPPIE/ponics", 0.26, 15 * tsz, MUTED)])
    T, M = int(TITLE_S * FPS), int(2.5 * FPS)
    seg_train = (T, T + hours)
    seg_mid = (seg_train[1], seg_train[1] + M)
    seg_unseen = (seg_mid[1], seg_mid[1] + hours)
    total = seg_unseen[1] + int(END_S * FPS) + FPS if max_frames is None else max_frames
    enc = _encoder(out, W, H)
    for k in range(total):
        if seg_train[0] <= k < seg_train[1]:
            s, h = "train", k - seg_train[0]
        elif seg_unseen[0] <= k < seg_unseen[1]:
            s, h = "unseen", k - seg_unseen[0]
        else:
            s, h = None, None
        cover.set_visible(s is None)
        suite_lbl.set_visible(s is not None)
        clock.set_visible(s is not None)
        for t in title:
            t.set_visible(k < T)
        for t in mid:
            t.set_visible(seg_mid[0] <= k < seg_mid[1])
        for t in end:
            t.set_visible(k >= seg_unseen[1])
        for su, (pa, ea, up_p, up_e) in ups.items():
            pa.set_visible(su == s)
            ea.set_visible(su == s)
        if s is not None:
            for nm in names:
                tanks[nm].update(data["runs"][s][nm], h)
            ups[s][2](h)
            ups[s][3](h)
            suite_lbl.set_text(f"ponics · {SUITE_TXT[s]}")
            clock.set_text(f"day {h // 24:2d}  {h % 24:02d}:00")
        _emit(fig, enc)
    enc.stdin.close()
    enc.wait()
    plt.close(fig)
    return out


def _job(args):
    kind, name, aspect, data_path, out = args
    data = json.loads(Path(data_path).read_text())
    if kind == "overview":
        return str(render_overview(data, Path(out), aspect))
    return str(render_controller(data, name, Path(out), aspect))


def main(argv=None) -> int:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser(prog="ponics-video")
    ap.add_argument("--data", default=str(ROOT / "results" / "replay" / "replay.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "videos"))
    ap.add_argument("--controller", action="append", default=[])
    ap.add_argument("--overview", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--aspect", choices=["wide", "square", "both"], default="both")
    a = ap.parse_args(argv)
    if not Path(a.data).exists():
        print(f"{a.data} missing: run `ponics-replay` first")
        return 2
    data = json.loads(Path(a.data).read_text())
    names = data["controllers"] if a.all else a.controller
    aspects = ["wide", "square"] if a.aspect == "both" else [a.aspect]
    jobs = [("controller", n, asp, a.data, f"{a.out}/{n}_{asp}.mp4") for n in names for asp in aspects]
    if a.all or a.overview:
        jobs += [("overview", None, asp, a.data, f"{a.out}/overview_{asp}.mp4") for asp in aspects]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 8)) as ex:
        for p in ex.map(_job, jobs):
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
