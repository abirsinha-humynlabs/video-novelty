"""The Signature: everything we keep about one span of one video.

A Signature is deliberately NOT a single vector. The whole argument of this
repo is that "same environment" and "same task" are different questions, so a
signature carries two independent descriptions plus a non-semantic fingerprint:

    tier 0  phash / phash_ts     -- did these files share literal footage?
    tier 1  app_frames / mean / var -- where is this?
    tier 2  motion_* / rhythm / period -- what work is happening?

Collapsing these into one number is the thing that makes thresholds arbitrary.
Keep them apart until the very last step (``metrics.fuse``), where the fusion
is explicit and swappable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .config import Config
from .encoders.appearance import static_weight_map
from .encoders.base import get_appearance_encoder
from .encoders.hashing import video_phashes
from .encoders.motion import FlowMotionEncoder
from .io.decode import VideoMeta, probe, sample_uniform

SCHEMA_VERSION = 1


def _video_id(path: str) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass
class Signature:
    # identity -----------------------------------------------------------
    segment_id: str
    video_id: str
    path: str
    t0: float
    t1: float
    # tier 1 -------------------------------------------------------------
    app_frames: np.ndarray                     # (N, D) float32, L2-normed rows
    app_mean: np.ndarray                       # (D,)
    app_var: np.ndarray                        # (D,)
    app_encoder: str = "gist"
    # tier 2 -------------------------------------------------------------
    motion_pooled: Optional[np.ndarray] = None    # (2*FRAME_DIM,)
    motion_seq: Optional[np.ndarray] = None       # (T', FRAME_DIM)
    rhythm: Optional[np.ndarray] = None           # (24,)
    motion_energy: Optional[np.ndarray] = None    # (T,)
    period_s: float = 0.0
    period_strength: float = 0.0
    ego_magnitude: float = 0.0
    # tier 0 -------------------------------------------------------------
    phash: Optional[np.ndarray] = None            # (K,) uint64
    phash_ts: Optional[np.ndarray] = None         # (K,) float
    # bookkeeping --------------------------------------------------------
    meta: Dict[str, Any] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    @property
    def duration(self) -> float:
        return max(self.t1 - self.t0, 0.0)

    @property
    def label(self) -> str:
        base = os.path.basename(self.path)
        if self.meta.get("whole_file"):
            return base
        return f"{base}[{self.t0:.0f}-{self.t1:.0f}s]"

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        arrays = {
            "app_frames": self.app_frames.astype(np.float16),
            "app_mean": self.app_mean.astype(np.float32),
            "app_var": self.app_var.astype(np.float32),
        }
        for name in ("motion_pooled", "motion_seq", "rhythm", "motion_energy", "phash", "phash_ts"):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = v
        header = {
            "segment_id": self.segment_id, "video_id": self.video_id, "path": self.path,
            "t0": self.t0, "t1": self.t1, "app_encoder": self.app_encoder,
            "period_s": self.period_s, "period_strength": self.period_strength,
            "ego_magnitude": self.ego_magnitude, "meta": self.meta, "schema": self.schema,
        }
        np.savez_compressed(path, header=np.frombuffer(json.dumps(header).encode(), np.uint8), **arrays)

    @classmethod
    def load(cls, path: str) -> "Signature":
        z = np.load(path, allow_pickle=False)
        header = json.loads(bytes(z["header"]).decode())
        kw: Dict[str, Any] = dict(header)
        kw["app_frames"] = z["app_frames"].astype(np.float32)
        kw["app_mean"] = z["app_mean"]
        kw["app_var"] = z["app_var"]
        for name in ("motion_pooled", "motion_seq", "rhythm", "motion_energy", "phash", "phash_ts"):
            kw[name] = z[name] if name in z.files else None
        return cls(**kw)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def _windows(meta: VideoMeta, cfg: Config):
    """Yield ``(t0, t1, is_whole_file)`` spans to sign.

    ``drop_last_shorter_than`` applies ONLY to a short trailing window, never
    to a full-length one. Getting that wrong silently reduces every file to a
    single signature whenever the window is shorter than the threshold, which
    looks like "windowing is not helping" rather than like a bug.
    """
    seg = cfg.segment
    if seg.seconds <= 0 or seg.seconds >= meta.duration:
        yield 0.0, meta.duration, True
        return
    hop = seg.hop if seg.hop > 0 else seg.seconds
    t = 0.0
    eps = 1e-6
    while t < meta.duration - eps:
        t1 = min(t + seg.seconds, meta.duration)
        span = t1 - t
        is_full = span >= seg.seconds - eps
        if is_full or span >= seg.drop_last_shorter_than or t == 0.0:
            yield t, t1, False
        t += hop


def build_signature(
    path: str,
    *,
    cfg: Optional[Config] = None,
    t0: float = 0.0,
    t1: Optional[float] = None,
    meta: Optional[VideoMeta] = None,
    appearance_encoder=None,
    whole_file: bool = True,
) -> Signature:
    """Build one Signature for ``path[t0:t1]``."""
    cfg = cfg or Config()
    meta = meta or probe(path)
    t1 = meta.duration if t1 is None else t1
    ac = cfg.appearance

    enc = appearance_encoder or get_appearance_encoder(ac.encoder, **ac.encoder_kwargs)
    frames, _ts = sample_uniform(
        path, n_frames=ac.n_frames, width=ac.width, height=ac.height,
        t0=t0, t1=t1, letterbox=ac.letterbox, meta=meta,
    )
    weights = None
    if ac.static_mask and getattr(enc, "supports_weights", False):
        weights = static_weight_map(
            frames, percentile=ac.static_percentile, floor=ac.static_floor,
            bottom_crop=ac.bottom_crop,
        )
    app = enc.encode(frames, weights)

    sig = Signature(
        segment_id=f"{_video_id(path)}:{int(round(t0))}-{int(round(t1))}",
        video_id=_video_id(path), path=os.path.abspath(path), t0=float(t0), t1=float(t1),
        app_frames=app.astype(np.float32), app_mean=app.mean(0).astype(np.float32),
        app_var=app.var(0).astype(np.float32), app_encoder=enc.name,
        meta={
            "whole_file": bool(whole_file),
            "source_duration": meta.duration,
            "source_fps": meta.fps,
            "source_size": [meta.width, meta.height],
            "codec": meta.codec,
            "static_mask": bool(weights is not None),
        },
    )

    if cfg.motion.enabled:
        mc = cfg.motion
        mf = FlowMotionEncoder(
            fps=mc.fps, width=mc.width, height=mc.height,
            max_seq=mc.max_seq, backend=mc.backend, letterbox=mc.letterbox,
            smooth_s=mc.smooth_s,
        ).encode(path, t0=t0, t1=t1)
        sig.motion_pooled = mf.pooled
        sig.motion_seq = mf.seq
        sig.rhythm = mf.rhythm
        sig.motion_energy = mf.energy.astype(np.float32)
        sig.period_s = mf.period_s
        sig.period_strength = mf.period_strength
        sig.ego_magnitude = mf.ego_magnitude
        if mc.clip_encoder:
            from .encoders.video import get_task_encoder
            te = get_task_encoder(mc.clip_encoder, **mc.clip_encoder_kwargs)
            sig.motion_pooled = te.encode(path, t0=t0, t1=t1)
            sig.meta["clip_encoder"] = mc.clip_encoder

    if cfg.hashing.enabled:
        h, hts = video_phashes(path, fps=cfg.hashing.fps, t0=t0, t1=t1)
        sig.phash, sig.phash_ts = h, hts

    return sig


def build_signatures(path: str, *, cfg: Optional[Config] = None,
                     appearance_encoder=None) -> List[Signature]:
    """Build one Signature per configured window (or a single whole-file one)."""
    cfg = cfg or Config()
    meta = probe(path)
    enc = appearance_encoder or get_appearance_encoder(
        cfg.appearance.encoder, **cfg.appearance.encoder_kwargs
    )
    out: List[Signature] = []
    for t0, t1, whole in _windows(meta, cfg):
        out.append(build_signature(path, cfg=cfg, t0=t0, t1=t1, meta=meta,
                                   appearance_encoder=enc, whole_file=whole))
    return out
