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
    """Tier 0: did these two clips share literal footage?

    ``min_overlap_fraction`` is what stops DUPLICATE_SOURCE from firing on
    repetitive work, and it exists because the obvious fixes do not work.
    Machine-paced manual work at a fixed bench repeats a ~5 s cycle with the
    same head pose, so two genuinely disjoint chunks contain matching runs of
    frames that are contiguous, consistently offset, and internally varying --
    every property that normally distinguishes real shared footage. Measured on
    the automobile corpus: disjoint chunks minutes apart matched for 2-5 s.
    pHash simply cannot separate "same footage" from "same action, same
    viewpoint, performed again" in this domain.

    What it CAN separate is scale. Actually shared footage -- a re-encode, or
    two cuts made with overlapping spans -- shares most of the shorter clip.
    Coincidental repetition shares a small slice of it (6-33% in the measured
    cases). So the DECISION requires a substantial fraction, while detection
    stays sensitive and the measured overlap is still reported in the verdict.
    That matters because DUPLICATE_SOURCE overrides every semantic axis, so it
    should only fire when the clips really are substantially the same.
    """
    enabled: bool = True
    fps: float = 2.0
    max_hamming: int = 8
    min_overlap_seconds: float = 2.0
    min_overlap_fraction: float = 0.5


@dataclass
class DecisionConfig:
    """Operating points, expressed as percentiles of the corpus null distribution.

    Read that twice: they are NOT raw cosine thresholds. "env_percentile: 52"
    means "closer in appearance than 52% of random pairs drawn from YOUR
    corpus". That is the only way a threshold survives moving from a factory
    dataset to a kitchen dataset. See docs/04-calibration.md.

    The right value tracks how redundant your corpus already is. A percentile
    threshold asks "is this pair in the top (100-p)% of my corpus", so it only
    means "same environment" when same-environment pairs are about that rare.
    That value moves when the corpus does, and has already moved once: fitted on
    two Pipe_Factory sessions (49% same-environment pairs) it came out at 52;
    adding an Automobile_Manufacturing session (41.7% same-environment) moved it
    to 60. At 52 against three environments, 11.3% of genuinely different pairs
    were called same-place. Re-fit whenever you add sessions -- the threshold
    sweep in `calibrate --labels` prints the best-F1 point.

    60 is best-F1 (0.969) over 741 pairs from three sessions. Zero-false-positive
    on that corpus is 62.5, which still keeps 90% of true redundants -- worth
    preferring, because the two errors are not symmetric: a false REDUNDANT
    DROPS footage and is unrecoverable, a false NOVEL only keeps something you
    did not need.

    use_task_axis is OFF, which is a deliberate retreat from the quadrant model.
    Measured on the same corpus, the task axis is strictly the weaker
    discriminator (best F1 0.905 vs 0.969; zero-FP threshold keeps 47% of true
    redundants vs 90%), correlates +0.82 with the environment axis, and orders
    domains WRONG -- it rates an automobile plant more task-similar to pipe
    factory A than pipe factory B is. Letting a detector that behaves like that
    gate the verdict produces confident, wrong labels such as
    SAME_PLACE_NEW_TASK for two unrelated industries.

    The cost is real and should not be forgotten: with the task axis off, two
    DIFFERENT jobs filmed in the SAME room both read REDUNDANT, and those are
    precisely the samples a physical-AI dataset is short of. We give that up
    because we currently cannot detect it, not because it stopped mattering.
    Turn this back on when a SAME_PLACE_NEW_TASK eval set exists (WORK.md s6)
    and the task axis clears a measured bar on it.
    """
    env_percentile: float = 60.0
    task_percentile: float = 95.0
    use_task_axis: bool = False


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
