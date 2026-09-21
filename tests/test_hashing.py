import numpy as np
from novelty.encoders.hashing import hamming, phash_frames, source_overlap, video_phashes


def test_phash_is_stable_under_reencode(clips, tmp_path):
    import subprocess
    out = tmp_path / "reenc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clips["a1"],
                    "-vf", "scale=96:72", "-crf", "35", str(out)], check=True)
    h1, _ = video_phashes(clips["a1"], fps=2)
    h2, _ = video_phashes(str(out), fps=2)
    d = np.diag(hamming(h1, h2[: len(h1)]))
    assert d.mean() < 12, "perceptual hash should survive a heavy re-encode + resize"


def test_source_overlap_detects_shared_footage(clips):
    h, t = video_phashes(clips["a1"], fps=4)
    half = len(h) // 2
    # two overlapping spans of the SAME recording
    ov = source_overlap(h[: half + 4], t[: half + 4], h[half - 4:], t[half - 4:])
    assert ov.overlap_seconds > 0
    assert ov.matched_frames >= 3


def test_source_overlap_rejects_different_videos(clips):
    h1, t1 = video_phashes(clips["a1"], fps=4)
    h2, t2 = video_phashes(clips["b2"], fps=4)
    ov = source_overlap(h1, t1, h2, t2)
    assert ov.overlap_seconds < 1.0


def test_disjoint_halves_are_not_source_duplicates(clips):
    """The user's own case: one file cut in two is NOT a file-level duplicate.

    Tier 0 must say 'no shared footage' so that tiers 1/2 can say 'same place,
    same work' without the two claims being confused.
    """
    h, t = video_phashes(clips["a1"], fps=4)
    half = len(h) // 2
    ov = source_overlap(h[:half], t[:half], h[half:], t[half:])
    assert ov.overlap_seconds < 1.0


def test_scattered_matches_are_not_shared_footage():
    """Repetitive work at a fixed bench produces coincidental frame matches.

    Real case this guards: two DISJOINT 30 s chunks of one automobile-plant
    recording were labelled DUPLICATE_SOURCE off four scattered near-identical
    frames that happened to share an offset bin. On repetitive manual work with
    a near-static head pose, individual frames seconds apart genuinely are the
    same pixels -- what never happens without shared footage is a continuous
    RUN of them. Overlap is measured as the longest contiguous run for exactly
    this reason, and DUPLICATE_SOURCE overrides every semantic axis, so a false
    positive here silently discards novel footage.
    """
    rng = np.random.default_rng(0)
    n = 60
    ha = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    hb = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    ta = tb = np.arange(n) * 0.5
    # five isolated frames match at a constant offset of 0 -- scattered, never
    # two in a row.
    for i in (3, 11, 24, 37, 52):
        hb[i] = ha[i]
    ov = source_overlap(ha, ta, hb, tb)
    assert ov.overlap_seconds < 2.0, (
        f"scattered matches must not read as shared footage, got {ov}")

    # the same number of matching frames, but CONTIGUOUS, is shared footage
    hb2 = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    hb2[20:25] = ha[20:25]
    ov2 = source_overlap(ha, ta, hb2, tb)
    assert ov2.overlap_seconds >= 2.0, f"a contiguous run must be detected, got {ov2}"


def test_phash_bits():
    g = np.zeros((2, 32, 32), np.uint8)
    g[1] = 255
    h = phash_frames(g)
    assert h.dtype == np.uint64 and len(h) == 2
