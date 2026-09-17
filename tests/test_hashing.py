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


def test_phash_bits():
    g = np.zeros((2, 32, 32), np.uint8)
    g[1] = 255
    h = phash_frames(g)
    assert h.dtype == np.uint64 and len(h) == 2
