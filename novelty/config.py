"""Configuration objects.

Everything tunable lives here so that an experiment is a YAML diff, not a code
diff. ``configs/*.yaml`` ship three presets:

    cpu.yaml   -- runs anywhere, no torch, no weights (the ``gist`` encoder)
    gpu.yaml   -- DINOv2 appearance + denser motion sampling
    default.yaml -- symlink-ish copy of cpu.yaml so `novelty` works out of the box
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml


@dataclass
class AppearanceConfig:
    encoder: str = "gist"
    n_frames: int = 48           # frames sampled per signature span
    width: int = 384             # decode size; the encoder resizes again internally
    height: int = 216
    letterbox: bool = True
    static_mask: bool = True     # egocentric body suppression (docs/03)
    static_percentile: float = 20.0
    static_floor: float = 0.15
    bottom_crop: float = 0.0     # extra hard crop, e.g. 0.2 kills the bottom fifth
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionConfig:
    enabled: bool = True
    fps: float = 10.0
    width: int = 160
    height: int = 90
    max_seq: int = 256
    backend: str = "farneback"
    letterbox: bool = True
    smooth_s: float = 0.5        # motion-energy smoothing window, seconds
    clip_encoder: Optional[str] = None   # None -> flow descriptor only.
                                         # "vjepa2" / "videomae" -> learned clip
                                         # embedding REPLACES motion_pooled while
                                         # flow still supplies rhythm + period.
    clip_encoder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashConfig:
    enabled: bool = True
    fps: float = 2.0
    max_hamming: int = 8
    min_overlap_seconds: float = 2.0


@dataclass
class DecisionConfig:
    """Operating points, expressed as percentiles of the corpus null distribution.

    Read that twice: they are NOT raw cosine thresholds. "env_percentile: 97"
    means "closer in appearance than 97% of random pairs drawn from YOUR
    corpus". That is the only way a threshold survives moving from a factory
    dataset to a kitchen dataset. See docs/04-calibration.md.
    """
    env_percentile: float = 97.0
    task_percentile: float = 95.0


@dataclass
class SegmentConfig:
    """How a long file is chopped before signing.

    ``seconds <= 0`` signs the whole file as one unit. For anything longer than
    ~2 minutes you almost certainly want windows: a 10-minute recording is not
    one homogeneous "thing", and a single pooled vector for it is a lie you
    will later mistake for a measurement.
    """
    seconds: float = 0.0
    hop: float = 0.0             # 0 -> non-overlapping (hop = seconds)
    drop_last_shorter_than: float = 5.0


@dataclass
class Config:
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    hashing: HashConfig = field(default_factory=HashConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Config":
        d = dict(d or {})
        sub = {
            "appearance": AppearanceConfig,
            "motion": MotionConfig,
            "hashing": HashConfig,
            "decision": DecisionConfig,
            "segment": SegmentConfig,
        }
        kw = {}
        for key, klass in sub.items():
            vals = d.get(key) or {}
            known = {f.name for f in dataclasses.fields(klass)}
            unknown = set(vals) - known
            if unknown:
                raise ValueError(f"unknown keys in config.{key}: {sorted(unknown)}")
            kw[key] = klass(**vals)
        unknown_top = set(d) - set(sub)
        if unknown_top:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown_top)}")
        return cls(**kw)

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        if not path:
            return cls()
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
