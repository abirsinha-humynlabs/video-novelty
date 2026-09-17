import numpy as np
from novelty.io.decode import iter_frames, probe, read_frames, sample_uniform


def test_probe(clips):
    m = probe(clips["a1"])
    assert m.width == 128 and m.height == 96
    assert 11.0 < m.duration < 13.0
    assert m.fps > 0


def test_iter_frames_streams_and_is_exhaustive(clips):
    seen = 0
    for chunk, ts in iter_frames(clips["a1"], fps=5, width=32, height=32, chunk=7):
        assert chunk.shape[1:] == (32, 32, 3)
        assert chunk.dtype == np.uint8
        assert len(ts) == len(chunk)
        seen += len(chunk)
    assert 55 <= seen <= 65          # ~12s at 5fps


def test_gray_and_letterbox(clips):
    f, _ = read_frames(clips["a1"], fps=2, width=64, height=64, gray=True, letterbox=True)
    assert f.shape[1:] == (64, 64, 1)
    # letterboxing a 4:3 source into a square must produce constant bars
    assert f[:, 0, :, 0].std() < 1.0 or f[:, :, 0, 0].std() < 1.0


def test_sample_uniform_count_and_span(clips):
    f, ts = sample_uniform(clips["a1"], n_frames=12, width=48, height=48)
    assert len(f) == 12
    assert ts[0] < ts[-1]


def test_time_window_is_respected(clips):
    _f, ts = read_frames(clips["a1"], fps=5, width=32, height=32, t0=2.0, t1=4.0)
    assert 1.9 <= ts[0] <= 2.2
    assert ts[-1] <= 4.3
