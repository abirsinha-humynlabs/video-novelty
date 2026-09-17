"""Synthetic fixtures.

The tests must run with no data and no network, so we render tiny videos with
ffmpeg: a "scene" is a coloured noise field plus a moving square, and a "task"
is the motion pattern of that square. Two clips can then share a scene but not
a task, or a task but not a scene -- which is exactly the 2x2 the pipeline is
supposed to resolve, and exactly what the shipped real-world eval set lacks.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest


def _render(path, frames, fps=10):
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
           "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.run(cmd, input=b"".join(f.astype(np.uint8).tobytes() for f in frames),
                       capture_output=True)
    assert p.returncode == 0, p.stderr.decode()[:400]
    return str(path)


def make_clip(path, *, scene_seed: int, period: float, seconds: float = 12.0,
              fps: int = 10, size=(128, 96), drift: float = 10.0, noise: float = 6.0,
              noise_seed: int = 0):
    """Render a synthetic clip.

    ``scene_seed`` picks the background texture (the "place"), ``period`` sets
    the motion cycle in seconds (the "task").

    The default 12 s length is not arbitrary either: autocorrelation needs
    roughly five or six repetitions before a period estimate stops being noise,
    so a 2 s cycle needs >= ~12 s of window. The same rule applies to real
    footage -- see docs/03-signatures.md.

    ``drift`` and ``noise`` are not decoration. Without them the clip is
    perfectly periodic, so a frame at t and a frame at t+period are literally
    the same pixels -- and a perceptual hash then (correctly!) reports that two
    disjoint halves of the file share footage. Real cameras drift and real
    sensors add noise, so the fixture does too. The lesson generalises: tier 0
    answers "are these the same pixels", and on exactly-looping synthetic
    content that is a different question from "is this the same file".
    """
    w, h = size
    rng = np.random.default_rng(scene_seed)
    n = int(seconds * fps)
    pan = int(drift * seconds) + 2
    bgw = w + pan
    bg = rng.integers(20, 200, size=(h, bgw, 3), dtype=np.uint8)
    bg = np.repeat(np.repeat(bg[::8, ::8], 8, 0), 8, 1)[:h, :bgw]
    nrng = np.random.default_rng(1000 + noise_seed)
    frames = []
    for i in range(n):
        off = int(drift * (i / fps))
        f = bg[:, off:off + w].copy().astype(np.int16)
        phase = 2 * np.pi * (i / fps) / period
        cx = int(w / 2 + (w / 4) * np.sin(phase))
        cy = int(h / 2 + (h / 6) * np.cos(phase))
        f[max(cy - 8, 0):cy + 8, max(cx - 8, 0):cx + 8] = 250
        f = f + nrng.normal(0, noise, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return _render(path, frames, fps)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {
        # same scene, same motion -> should be the most similar pair
        "a1": make_clip(d / "a1.mp4", scene_seed=1, period=2.0, noise_seed=1),
        "a2": make_clip(d / "a2.mp4", scene_seed=1, period=2.0, noise_seed=2),
        # same scene, different motion -> SAME_PLACE_NEW_TASK
        "a3": make_clip(d / "a3.mp4", scene_seed=1, period=5.0, noise_seed=3),
        # different scene, same motion -> SAME_TASK_NEW_PLACE
        "b1": make_clip(d / "b1.mp4", scene_seed=99, period=2.0, noise_seed=4),
        # different scene, different motion -> NOVEL
        "b2": make_clip(d / "b2.mp4", scene_seed=99, period=5.0, noise_seed=5),
    }
