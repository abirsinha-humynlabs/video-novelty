"""novelty -- measure how much NEW information a video adds to a corpus.

The public surface is deliberately small:

    from novelty import Signature, build_signature, Index, compare

Everything else lives behind ``novelty.<subpackage>`` and is documented in
``docs/``.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .signature import Signature, build_signature          # noqa: E402,F401
from .store.numpy_store import Index                        # noqa: E402,F401
from .metrics.fuse import compare, Verdict                   # noqa: E402,F401

__all__ = ["Signature", "build_signature", "Index", "compare", "Verdict", "__version__"]
