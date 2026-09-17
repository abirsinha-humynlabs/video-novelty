"""Appearance-encoder registry.

Every appearance encoder is a callable object with:

    .name  : str
    .dim   : int
    .encode(frames_uint8_rgb, weights=None) -> (T, dim) float32, L2-normalised rows

``weights`` is an optional (T, H, W) float map in [0, 1] letting the caller
suppress regions -- used by the egocentric static-body mask (see docs/03).
Encoders that cannot honour a spatial weight map may ignore it, but must say
so via ``.supports_weights``.

Swapping backbones is the single highest-leverage knob in this repo, which is
why it is a registry and not an import.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol

import numpy as np

_REGISTRY: Dict[str, Callable[..., "AppearanceEncoder"]] = {}


class AppearanceEncoder(Protocol):
    name: str
    dim: int
    supports_weights: bool

    def encode(self, frames: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        ...


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def list_appearance_encoders():
    return sorted(_REGISTRY)


def get_appearance_encoder(name: str, **kw) -> AppearanceEncoder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown appearance encoder {name!r}; have {list_appearance_encoders()}")
    return _REGISTRY[name](**kw)


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)
