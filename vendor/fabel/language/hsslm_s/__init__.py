"""
hsslm_s — symbolic language-form modules (slim bundle).

This is the minimal subset the fabel conversation layer needs to turn graph
facts into measured sentence form. The full symbolic package (Möbius core,
Mac-M4 optimizers, parallel engine, Foss gate, ...) lives in the SYMBOLISCH
project; only these five weight-free modules are bundled here:

  sampler       — tau-controlled contraction sampler (connective choice)
  state_init    — hyperboloid symbol states (Berry-phase repetition guard)
  pattern_bank  — mined sentence-form patterns + F30 weak-signal backoff
  inference     — Jaro-Winkler, Möbius confidence (entity resolution)
  bphm          — Berry-phase repetition detection

Everything here is deterministic and dependency-light (NumPy only).
"""
from __future__ import annotations

from . import sampler, state_init, pattern_bank, inference, bphm

__all__ = ["sampler", "state_init", "pattern_bank", "inference", "bphm"]
