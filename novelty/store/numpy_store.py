"""On-disk index: a directory of .npz signatures plus JSON sidecars.

Layout::

    <index>/
      config.yaml          the exact config used to build every signature here
      manifest.json        id -> {path, t0, t1, encoder, ...}
      null.json            fitted null model (written by `novelty calibrate`)
      signatures/<id>.npz

Why not a vector database on day one: at 1e4-1e5 signatures a brute-force
float32 matmul is milliseconds and costs zero operational surface. The store
interface below is small on purpose so that swapping in FAISS/Qdrant later is
a ~50-line adapter -- see docs/02-architecture.md for the crossover point.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..signature import Signature


class Index:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.sig_dir = os.path.join(self.root, "signatures")
        self._sigs: Optional[List[Signature]] = None

    # -- lifecycle -------------------------------------------------------
    def create(self, cfg: Config) -> "Index":
        os.makedirs(self.sig_dir, exist_ok=True)
        cfg.dump(os.path.join(self.root, "config.yaml"))
        if not os.path.exists(self.manifest_path):
            self._write_manifest({})
        return self

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    @property
    def null_path(self) -> str:
        return os.path.join(self.root, "null.json")

    @property
    def config_path(self) -> str:
        return os.path.join(self.root, "config.yaml")

    def config(self) -> Config:
        return Config.load(self.config_path) if os.path.exists(self.config_path) else Config()

    def manifest(self) -> Dict[str, dict]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path) as fh:
            return json.load(fh)

    def _write_manifest(self, m: Dict[str, dict]) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.manifest_path, "w") as fh:
            json.dump(m, fh, indent=2)

    # -- writes ----------------------------------------------------------
    def add(self, sigs: Iterable[Signature]) -> List[str]:
        os.makedirs(self.sig_dir, exist_ok=True)
        m = self.manifest()
        added = []
        for s in sigs:
            fn = os.path.join(self.sig_dir, f"{s.segment_id.replace(':', '_')}.npz")
            s.save(fn)
            m[s.segment_id] = {
                "file": os.path.relpath(fn, self.root), "path": s.path,
                "video_id": s.video_id, "t0": s.t0, "t1": s.t1,
                "label": s.label, "encoder": s.app_encoder,
                "period_s": s.period_s, "period_strength": s.period_strength,
                "ego_magnitude": s.ego_magnitude,
            }
            added.append(s.segment_id)
        self._write_manifest(m)
        self._sigs = None
        return added

    # -- reads -----------------------------------------------------------
    def signatures(self, reload: bool = False) -> List[Signature]:
        if self._sigs is not None and not reload:
            return self._sigs
        m = self.manifest()
        out = []
        for sid, row in sorted(m.items()):
            fn = os.path.join(self.root, row["file"])
            if os.path.exists(fn):
                out.append(Signature.load(fn))
        self._sigs = out
        return out

    def labels(self) -> List[str]:
        return [s.label for s in self.signatures()]

    def __len__(self) -> int:
        return len(self.manifest())

    # -- null model ------------------------------------------------------
    def null(self):
        from ..calibrate import NullModel
        if os.path.exists(self.null_path):
            return NullModel.load(self.null_path)
        return None

    def save_null(self, null) -> None:
        null.save(self.null_path)

    # -- similarity ------------------------------------------------------
    def similarity_matrix(self, kind: str = "fused", null=None) -> np.ndarray:
        """Dense pairwise similarity over every signature in the index.

        ``kind``: ``env`` | ``task`` | ``fused``. Values are calibrated
        percentiles scaled to [0, 1] when a null model is available, which is
        what makes them safe to feed to the submodular selector -- raw cosines
        squashed into [0.9, 1.0] make every clip look equally well covered.
        """
        from ..metrics.fuse import raw_scores
        sigs = self.signatures()
        n = len(sigs)
        S = np.eye(n, dtype=np.float32)
        null = null if null is not None else self.null()
        wh = null.whitener if null is not None else None
        for i in range(n):
            for j in range(i + 1, n):
                r = raw_scores(sigs[i], sigs[j], whitener=wh,
                               weights=(null.weights if null is not None else None))
                if null is not None:
                    env = null.percentile("env_raw", r["env_raw"]) / 100.0
                    task = null.percentile("task_raw", r["task_raw"]) / 100.0
                else:
                    env = (r["env_raw"] + 1) / 2
                    task = (r["task_raw"] + 1) / 2
                v = {"env": env, "task": task, "fused": 0.5 * (env + task)}[kind]
                S[i, j] = S[j, i] = np.float32(v)
        return S
