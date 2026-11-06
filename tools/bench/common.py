"""Shared paths for the bench scripts. Override with environment variables on
another machine (Windows PowerShell:  $env:STACKER_FRAMES = "R:\\TIFF").

  STACKER_FRAMES  folder with the source frames (and Helicon outputs)
  STACKER_GLOB    frame filename pattern (default IMG_*.tif)
  STACKER_OUT     analysis/output folder (default <repo>/out/tiff_2026-11-05)
  STACKER_STEP_UM rail step in um (default 0.468)   STACKER_FOV_MM (default 0.98)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))

FRAMES = Path(os.environ.get("STACKER_FRAMES", str(Path.home() / "Desktop" / "TIFF")))
GLOB = os.environ.get("STACKER_GLOB", "IMG_*.tif")
OUT = Path(os.environ.get("STACKER_OUT", str(REPO / "out" / "tiff_2026-11-05")))
STEP_UM = float(os.environ.get("STACKER_STEP_UM", "0.468"))
FOV_MM = float(os.environ.get("STACKER_FOV_MM", "0.98"))
FUSION = OUT / "fusion_0.5"

# 1:1 comparison regions used throughout docs/FINDINGS.md (half-res grid of the
# 2026-11-05 tyuyamunite stack; reference-frame full-res ROIs for roi_compare)
HALF_RES_ROIS = [(1700, 950, "crystal fan"), (650, 1500, "crystal edge"),
                 (1000, 1650, "matrix grains"), (2200, 720, "specimen / background edge")]
FULL_RES_ROIS = {"plates": (3500, 3000), "fan": (3380, 1950), "crystal_edge": (1250, 2950),
                 "bg_edge": (4229, 1328)}


def helicon(kind: str):
    """Helicon output in FRAMES: kind B, C or comp (matched by Helicon's '(B,' naming)."""
    m = sorted(FRAMES.glob(f"*({kind},*.tif"))
    return m[0] if m else None
