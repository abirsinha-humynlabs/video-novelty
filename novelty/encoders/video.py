"""Optional learned clip encoders for tier 2 (the GPU path).

The hand-crafted flow descriptor in ``motion.py`` is honest but blunt: it sees
"how much stuff moved, where, in which direction, how rhythmically". It does
not see *what* is being manipulated, and it cannot tell "picking a fitting out
of a crate" from "putting a fitting into a crate" -- reversed actions produce
mirrored flow, which the orientation histogram does distinguish, but subtler
semantic differences are beyond it.

A self-supervised video backbone sees those. Two are wired up:

``vjepa2``    V-JEPA 2 -- trained by predicting masked spatio-temporal latents.
              Currently the strongest general motion representation for
              robotics-adjacent video, and the one to reach for first on a GPU
              box.
``videomae``  VideoMAE v2 -- older, smaller, faster, well-understood. A good
              fallback and a good sanity check: if your results change wildly
              between the two, the signal you are seeing is a backbone
              artefact, not a property of your data.

Both are run over sliding 16-frame windows and mean-pooled, then L2-normalised
so they drop straight into the same cosine/whitening machinery as the flow
descriptor. The flow encoder still runs alongside, because rhythm and cycle
period are cheap, interpretable and NOT recoverable from a pooled clip
embedding -- a 16-frame window is ~0.5 s and cannot see a 6-second work cycle.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..io.decode import iter_frames

_TASK_REGISTRY: Dict[str, type] = {}


def register_task_encoder(name: str):
    def deco(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return deco


def list_task_encoders() -> List[str]:
    return sorted(_TASK_REGISTRY)


def get_task_encoder(name: str, **kw):
    if name not in _TASK_REGISTRY:
        raise KeyError(f"unknown task encoder {name!r}; have {list_task_encoders()}")
    return _TASK_REGISTRY[name](**kw)


class _HFVideoEncoder:
    """Sliding-window wrapper around a HuggingFace video backbone."""

    def __init__(self, model_id: str, name: str, *, clip_len: int = 16, stride: int = 8,
                 fps: float = 8.0, size: int = 224, device: Optional[str] = None,
                 batch_size: int = 4, dtype: str = "float16"):
        self.model_id = model_id
        self.name = name
        self.clip_len = clip_len
        self.stride = stride
        self.fps = fps
        self.size = size
        self.batch_size = batch_size
        self._dtype_name = dtype
        self._device = device
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"task encoder {self.name!r} needs torch + transformers:\n"
                f"    pip install 'novelty[gpu]'"
            ) from exc
        self._torch = torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        dt = getattr(torch, self._dtype_name) if dev == "cuda" else torch.float32
        try:
            self._proc = AutoVideoProcessor.from_pretrained(self.model_id)
        except Exception:                                   # noqa: BLE001
            from transformers import AutoImageProcessor
            self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, torch_dtype=dt).to(dev).eval()

    def _prepare(self, batch: List[np.ndarray]):
        """Windows -> model inputs. Override when the processor signature differs.

        ``batch`` is a list of ``(clip_len, H, W, 3)`` uint8 arrays.
        """
        inp = self._proc([list(w) for w in batch], return_tensors="pt").to(self._device)
        if self._model.dtype != self._torch.float32:
            inp = {k: (v.to(self._model.dtype) if v.is_floating_point() else v)
                   for k, v in inp.items()}
        return inp

    def _forward(self, inp):
        """Model inputs -> (B, D) float tensor. Override per-architecture."""
        return self._pool(self._model(**inp))

    def _pool(self, out):
        h = out.last_hidden_state              # (B, T*P, D)
        return h.mean(dim=1)

    def encode(self, path: str, *, t0: Optional[float] = None,
               t1: Optional[float] = None) -> np.ndarray:
        """-> (D,) L2-normalised clip embedding, mean-pooled over windows."""
        self._lazy()
        torch = self._torch
        frames, _ts = [], None
        buf: List[np.ndarray] = []
        windows: List[np.ndarray] = []
        for chunk, _t in iter_frames(path, fps=self.fps, width=self.size, height=self.size,
                                     t0=t0, t1=t1, letterbox=True, chunk=64):
            for f in chunk:
                buf.append(f)
                if len(buf) == self.clip_len:
                    windows.append(np.stack(buf))
                    buf = buf[self.stride:]
        if buf and not windows:
            pad = buf + [buf[-1]] * (self.clip_len - len(buf))
            windows.append(np.stack(pad))
        if not windows:
            return np.zeros(1, np.float32)

        feats = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                inp = self._prepare(windows[i:i + self.batch_size])
                feats.append(self._forward(inp).float().cpu().numpy())
        V = np.concatenate(feats, 0).mean(0).astype(np.float32)
        return V / (np.linalg.norm(V) + 1e-8)


@register_task_encoder("vjepa2")
class VJepa2Encoder(_HFVideoEncoder):
    def __init__(self, model_id: str = "facebook/vjepa2-vitl-fpc64-256", **kw):
        kw.setdefault("size", 256)
        kw.setdefault("clip_len", 16)
        super().__init__(model_id, "vjepa2", **kw)


@register_task_encoder("videomae")
class VideoMAEEncoder(_HFVideoEncoder):
    def __init__(self, model_id: str = "MCG-NJU/videomae-base", **kw):
        super().__init__(model_id, "videomae", **kw)


@register_task_encoder("cosmos-embed1")
class CosmosEmbed1Encoder(_HFVideoEncoder):
    """NVIDIA Cosmos-Embed1 -- a video/text joint embedder.

    Worth having because its training mix (AgiBot, BridgeV2, DROID, RoboNet, 1X)
    is robotics/egocentric rather than Kinetics-style action clips, so it has
    seen manipulation footage that looks like ours. Variants: ``224p`` -> 256-d,
    ``336p`` and ``448p`` -> 768-d.

    Two caveats that decide how you read its scores:

    * It is **text-aligned** (contrastive against captions). That is the same
      caption-collapse risk docs/01 cites for rejecting CLIP on the environment
      axis -- two different factories both captionable as "worker sorting parts"
      can converge. It is wired in on the *task* axis only, where that failure
      is less damaging, and it is an alternative to ``vjepa2``, not a
      replacement for the flow descriptor's rhythm/period features.
    * It pools **8 frames** per forward pass. At the default ``fps=2.0`` one
      window spans 4 s of wall-clock, so a 30 s segment is many windows
      mean-pooled -- it still cannot see a 6 s work cycle in one pass, which is
      exactly why motion.py's autocorrelation features keep running alongside.
    """

    def __init__(self, model_id: str = "nvidia/Cosmos-Embed1-336p", **kw):
        kw.setdefault("size", 336)
        kw.setdefault("clip_len", 8)      # the released variants are tuned for 8
        kw.setdefault("stride", 4)
        kw.setdefault("fps", 2.0)         # 8 frames @ 2 fps == 4 s of context
        kw.setdefault("dtype", "bfloat16")
        super().__init__(model_id, "cosmos-embed1", **kw)

    def _lazy(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"task encoder {self.name!r} needs torch + transformers:\n"
                f"    pip install 'novelty[gpu]'"
            ) from exc
        self._torch = torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        # bfloat16 on Ampere+; CPU gets float32 because bf16 matmul there is slow.
        self._dt = getattr(torch, self._dtype_name) if dev == "cuda" else torch.float32
        self._proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(dev, dtype=self._dt).eval()

    def _prepare(self, batch: List[np.ndarray]):
        # Cosmos wants BTCHW, not the BTHWC that decode gives us.
        arr = np.transpose(np.stack(batch), (0, 1, 4, 2, 3))
        return self._proc(videos=arr).to(self._device, dtype=self._dt)

    def _forward(self, inp):
        return self._model.get_video_embeddings(**inp).visual_proj
