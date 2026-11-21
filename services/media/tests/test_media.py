import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ponics_media.render import FPS, TITLE_S, frame_hour, n_frames, ph_color, render_controller

VERIFY = Path.home() / ".claude/skills/motion-studio/scripts/verify_media.sh"


def fake_replay(hours=48):
    t = np.arange(hours)
    run = {"ph": list(5.8 + 0.3 * np.sin(t / 5)), "ec": [1.8] * hours, "ph_meas": list(5.8 + 0.3 * np.sin(t / 5)),
           "ec_meas": [1.8] * hours, "acid": [2.0] * hours, "base": [0.0] * hours, "ab": [0.0] * hours,
           "fw": list(3 + t), "stress": [1.0] * hours, "dump_days": [1],
           "metrics": {"ph_in_band": 1.0, "ph_severe_h": 0.0, "fw_g": 50.0, "acid_l": 0.1, "base_l": 0.0, "ph_rms": 0.1}}
    return {"setpoints": {"ec": 1.8, "ph": 5.8, "ph_lo": 5.5, "ph_hi": 6.5, "ec_band": 0.2}, "hours": hours,
            "controllers": ["mpc"], "runs": {"train": {"mpc": run}, "unseen": {"mpc": run}}}


def test_timeline_is_a_pure_function_of_frame():
    assert frame_hour(0, 840) is None and frame_hour(int(TITLE_S * FPS), 840) == 0
    assert frame_hour(10_000, 840) == 839
    assert n_frames(840) == int(TITLE_S * FPS) + 840 + 4 * FPS


def test_indicator_colours_are_ordered():
    r4, g4, b4 = ph_color(4.0)
    r8, g8, b8 = ph_color(8.0)
    assert r4 > b4 and b8 > r8  # acid red, alkaline blue


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_smoke_clip(tmp_path):
    out = render_controller(fake_replay(), "mpc", tmp_path / "c.mp4", "square", max_frames=45)
    if VERIFY.exists():
        r = subprocess.run([str(VERIFY), str(out), "--width", "1080", "--height", "1080", "--duration", "1.5",
                            "--no-audio", "--faststart"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout
    else:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,codec_name",
                                "-of", "json", str(out)], capture_output=True, text=True)
        s = json.loads(probe.stdout)["streams"][0]
        assert (s["width"], s["height"], s["codec_name"]) == (1080, 1080, "h264")
