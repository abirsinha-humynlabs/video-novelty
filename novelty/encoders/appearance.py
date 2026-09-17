"""Tier 1 -- appearance / "is this the same physical place?"

Three encoders ship here:

``gist``     dependency-free (numpy + OpenCV). A multi-scale oriented-gradient
             + spatial-colour descriptor in the spirit of Oliva & Torralba's
             GIST. Weak compared to a learned backbone, but it needs no
             weights, no GPU and no network, so the repo runs end-to-end on a
             laptop and in CI. Use it for plumbing, not for production
             judgements.
``dinov2``   ViT self-supervised on images. The right default on a GPU box.
             DINOv2 beats CLIP here on purpose: CLIP is text-aligned, so it
             collapses visually distinct scenes that share a caption
             ("a factory"), which is the exact failure mode we are trying to
             avoid. DINOv2 keeps instance-level appearance.
``siglip``   text-aligned alternative, useful when you *also* want to query the
             index with words ("find me clips with a forklift").

All three return L2-normalised rows so cosine == dot product.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import register, l2norm

# --------------------------------------------------------------------------
# static-region (egocentric body) masking
# --------------------------------------------------------------------------

def static_weight_map(
    frames: np.ndarray,
    *,
    percentile: float = 20.0,
    floor: float = 0.15,
    bottom_crop: float = 0.0,
    smooth_sigma: float = 6.0,
) -> np.ndarray:
    """Down-weight pixels that never change across the clip.

    On a head-mounted camera the wearer's own torso, arms-at-rest and trouser
    legs occupy a large, *constant* part of the bottom of every frame the
    wearer ever records. Those pixels make two unrelated tasks look similar
    (you end up matching trousers, not environments). We estimate them from
    the data instead of hard-coding a crop: pixels whose temporal standard
    deviation falls below ``percentile`` get weight ``floor``, the rest get 1.

    Returns ``(T, H, W)`` float32 in [floor, 1] -- broadcast per-frame, since
    the mask is a property of the clip, not of any one frame.
    """
    if frames.ndim == 4 and frames.shape[-1] == 3:
        g = frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
    else:
        g = frames.astype(np.float32).squeeze(-1) if frames.ndim == 4 else frames.astype(np.float32)
    g = g.astype(np.float32)
    std = g.std(axis=0)
    if smooth_sigma > 0:
        std = cv2.GaussianBlur(std, (0, 0), smooth_sigma)
    thr = np.percentile(std, percentile)
    w = np.where(std <= thr, floor, 1.0).astype(np.float32)
    if bottom_crop > 0:
        h = w.shape[0]
        w[int(h * (1.0 - bottom_crop)):, :] = floor
    if smooth_sigma > 0:
        w = cv2.GaussianBlur(w, (0, 0), smooth_sigma)
    return np.broadcast_to(w[None], (len(g),) + w.shape).copy()


# --------------------------------------------------------------------------
# gist -- dependency-free fallback
# --------------------------------------------------------------------------

@register("gist")
class GistEncoder:
    """Multi-scale oriented-gradient energy + spatial colour histograms.

    Layout of the 564-d vector:
      384 = 3 scales x 4x4 spatial cells x 8 gradient orientations
      180 = 3x3 spatial cells x (12 hue + 4 sat + 4 val) bins
    Each block is L2-normalised on its own before the whole vector is, so
    texture and colour contribute comparably regardless of scene contrast.
    """

    name = "gist"
    supports_weights = True

    def __init__(self, size: int = 256, scales=(1.0, 2.0, 4.0), grid: int = 4,
                 orientations: int = 8, color_grid: int = 3):
        self.size = size
        self.scales = tuple(scales)
        self.grid = grid
        self.orientations = orientations
        self.color_grid = color_grid
        self.dim = len(self.scales) * grid * grid * orientations + color_grid * color_grid * 20

    # -- helpers ---------------------------------------------------------
    def _grad_block(self, gray: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
        b = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy) * w
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        idx = np.minimum((ang / np.pi * self.orientations).astype(np.int32), self.orientations - 1)
        n = self.grid
        cell = self.size // n
        out = np.empty((n, n, self.orientations), np.float32)
        for r in range(n):
            for c in range(n):
                sl = (slice(r * cell, (r + 1) * cell), slice(c * cell, (c + 1) * cell))
                out[r, c] = np.bincount(
                    idx[sl].ravel(), weights=mag[sl].ravel(), minlength=self.orientations
                ).astype(np.float32)
        return out.ravel()

    def _color_block(self, rgb: np.ndarray, w: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
        hb = np.minimum(h * 12 // 180, 11)
        sb = np.minimum(s * 4 // 256, 3)
        vb = np.minimum(v * 4 // 256, 3)
        n = self.color_grid
        cell = self.size // n
        out = np.empty((n, n, 20), np.float32)
        for r in range(n):
            for c in range(n):
                sl = (slice(r * cell, (r + 1) * cell), slice(c * cell, (c + 1) * cell))
                ww = w[sl].ravel()
                out[r, c, :12] = np.bincount(hb[sl].ravel(), weights=ww, minlength=12)
                out[r, c, 12:16] = np.bincount(sb[sl].ravel(), weights=ww, minlength=4)
                out[r, c, 16:20] = np.bincount(vb[sl].ravel(), weights=ww, minlength=4)
        return out.ravel()

    # -- api -------------------------------------------------------------
    def encode(self, frames: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(frames)
        out = np.empty((n, self.dim), np.float32)
        for i in range(n):
            rgb = cv2.resize(frames[i], (self.size, self.size), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            if weights is not None:
                w = cv2.resize(weights[i], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            else:
                w = np.ones((self.size, self.size), np.float32)
            blocks = [l2norm(self._grad_block(gray, w, s)) for s in self.scales]
            blocks.append(l2norm(self._color_block(rgb, w)))
            out[i] = np.concatenate(blocks)
        return l2norm(out, axis=1)


# --------------------------------------------------------------------------
# torch backbones (lazy -- importing this module must not require torch)
# --------------------------------------------------------------------------

class _TorchVisionEncoder:
    """Shared plumbing for HuggingFace image backbones."""

    supports_weights = True   # applied as a pixel-space soft mask before the ViT

    def __init__(self, model_id: str, name: str, dim: int, device: Optional[str] = None,
                 batch_size: int = 32, dtype: str = "float16"):
        self.name = name
        self.model_id = model_id
        self.dim = dim
        self.batch_size = batch_size
        self._dtype_name = dtype
        self._device = device
        self._model = None
        self._proc = None

    def _lazy(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"encoder {self.name!r} needs torch + transformers:\n"
                f"    pip install 'novelty[gpu]'"
            ) from exc
        self._torch = torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        dt = getattr(torch, self._dtype_name) if dev == "cuda" else torch.float32
        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id, torch_dtype=dt).to(dev).eval()

    def _pool(self, out):  # override per-architecture
        h = out.last_hidden_state
        return h[:, 0]      # CLS token

    def encode(self, frames: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        self._lazy()
        torch = self._torch
        if weights is not None:
            # soft-mask toward grey rather than black: a hard black rectangle is
            # itself a strong, spurious feature for a ViT.
            f = frames.astype(np.float32)
            w = weights[..., None]
            frames = (f * w + 128.0 * (1.0 - w)).astype(np.uint8)
        feats = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = list(frames[i:i + self.batch_size])
                inp = self._proc(images=batch, return_tensors="pt").to(self._device)
                if self._model.dtype != torch.float32:
                    inp = {k: (v.to(self._model.dtype) if v.is_floating_point() else v)
                           for k, v in inp.items()}
                feats.append(self._pool(self._model(**inp)).float().cpu().numpy())
        return l2norm(np.concatenate(feats, 0).astype(np.float32), axis=1)


@register("dinov2")
def _dinov2(model_id: str = "facebook/dinov2-base", **kw):
    return _TorchVisionEncoder(model_id, "dinov2", 768, **kw)


@register("dinov2-large")
def _dinov2l(model_id: str = "facebook/dinov2-large", **kw):
    return _TorchVisionEncoder(model_id, "dinov2-large", 1024, **kw)


@register("siglip")
def _siglip(model_id: str = "google/siglip-base-patch16-224", **kw):
    enc = _TorchVisionEncoder(model_id, "siglip", 768, **kw)

    def _pool(out):
        return out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(1)
    enc._pool = _pool  # type: ignore[method-assign]
    return enc
