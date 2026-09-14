"""
FibreBlend — vision branch.

Quantitative fibre-blend composition regression from RGB macro images of fabric.
This is the imaging counterpart of the NIR pipeline in ``src/`` and targets the
Phase-2 gap named in the repo README: blend-level accuracy from a cheap RGB
sensor instead of a lab NIR spectrometer.

Design rationale, trade-off matrix and loss derivation:
    docs/vision_architecture_decision.md
"""

from __future__ import annotations

__all__ = [
    "preprocessing",
    "fabric_simulator",
    "data",
    "model",
    "losses",
    "train",
]
