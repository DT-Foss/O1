"""Local shim: re-export the bundled canonical dotcausal package.

Keeps `from causal_format import CausalWriter, CausalReader, INFERENCE_CONFIG`
working without depending on anything outside this folder. The package lives
in ../dotcausal_package/src.
"""
import os
import sys

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "dotcausal_package", "src")
_PKG = os.path.abspath(_PKG)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from dotcausal.core import INFERENCE_CONFIG  # noqa: F401
from dotcausal.io import CausalWriter, CausalReader  # noqa: F401
