import numpy as np
import pytest

from novelty.config import Config
from novelty.signature import Signature, build_signature, build_signatures


def test_config_roundtrip(tmp_path):
    c = Config()
    c.appearance.n_frames = 17
    c.motion.fps = 7.5
    p = tmp_path / "c.yaml"
    c.dump(str(p))
    back = Config.load(str(p))
    assert back.appearance.n_frames == 17
    assert back.motion.fps == 7.5


def test_config_rejects_typos():
    with pytest.raises(ValueError, match="unknown keys"):
        Config.from_dict({"appearance": {"encodr": "gist"}})
    with pytest.raises(ValueError, match="unknown top-level"):
        Config.from_dict({"appearence": {}})


def test_build_signature_shapes(clips):
    cfg = Config()
    cfg.appearance.n_frames = 16
    s = build_signature(clips["a1"], cfg=cfg)
    assert s.app_frames.shape == (16, 564)
    assert np.allclose(np.linalg.norm(s.app_frames, axis=1), 1.0, atol=1e-4)
    assert s.app_mean.shape == (564,) and s.app_var.shape == (564,)
    assert s.motion_pooled is not None and s.motion_pooled.ndim == 1
    assert s.motion_seq.ndim == 2 and s.motion_seq.shape[1] == 131
    assert s.rhythm.shape == (24,)
    assert s.phash is not None and s.phash.dtype == np.uint64
    assert 0.0 < s.duration <= 13.0


def test_signature_save_load_roundtrip(clips, tmp_path):
    cfg = Config()
    cfg.appearance.n_frames = 12
    s = build_signature(clips["a1"], cfg=cfg)
    f = tmp_path / "s.npz"
    s.save(str(f))
    b = Signature.load(str(f))
    assert b.segment_id == s.segment_id
    assert b.app_frames.shape == s.app_frames.shape
    # app_frames are stored as float16 to keep indexes small
    assert np.allclose(b.app_mean, s.app_mean, atol=1e-6)
    assert np.allclose(b.rhythm, s.rhythm)
    assert b.period_s == pytest.approx(s.period_s)


def test_windowing_produces_multiple_segments(clips):
    cfg = Config()
    cfg.appearance.n_frames = 8
    cfg.segment.seconds = 2.0
    cfg.segment.hop = 2.0
    sigs = build_signatures(clips["a1"], cfg=cfg)
    assert len(sigs) >= 3
    assert all(s.video_id == sigs[0].video_id for s in sigs)
    assert len({s.segment_id for s in sigs}) == len(sigs)
    assert sigs[0].t1 == pytest.approx(2.0)


def test_detected_period_tracks_the_synthetic_cycle(clips):
    """The rhythm feature must recover the cycle we rendered in."""
    cfg = Config()
    cfg.appearance.n_frames = 8
    fast = build_signature(clips["a1"], cfg=cfg)   # 2.0 s cycle
    slow = build_signature(clips["a3"], cfg=cfg)   # 5.0 s cycle
    assert fast.period_s > 0 and slow.period_s > 0
    assert fast.period_s < slow.period_s


def test_static_mask_changes_the_embedding(clips):
    """The egocentric body mask must actually do something."""
    on, off = Config(), Config()
    on.appearance.n_frames = off.appearance.n_frames = 10
    off.appearance.static_mask = False
    a = build_signature(clips["a1"], cfg=on)
    b = build_signature(clips["a1"], cfg=off)
    assert not np.allclose(a.app_mean, b.app_mean, atol=1e-4)
