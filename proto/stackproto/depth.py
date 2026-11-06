"""Coarse depth map from a luma stack (focus-measure argmax, sub-frame refined).

Used by the fusion prototypes (A8). Note from the A6 investigation: depth-gated
alignment and per-pixel blur equalisation were both tried to remove the
specimen-relief bias in scale/drift and NEITHER worked -- the bias comes from
defocused light spreading in from neighbouring depths, which no per-pixel model
captures (see docs/FINDINGS.md).
"""
from __future__ import annotations

import cv2
import numpy as np


def focus_measure(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    lap = cv2.Laplacian(cv2.GaussianBlur(img, (0, 0), 0.7), cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(lap * lap, (0, 0), sigma)


def estimate_depth(stack, sigma: float = 2.0):
    """Streaming argmax of the focus measure with parabolic sub-frame refinement.

    Returns (depth_index float32, confidence float32 in ~[0, 1]).
    """
    n, h, w = stack.shape
    best = np.full((h, w), -1.0, np.float32)
    bestk = np.zeros((h, w), np.int32)
    before = np.zeros((h, w), np.float32)    # FM at bestk-1
    after = np.zeros((h, w), np.float32)     # FM at bestk+1
    total = np.zeros((h, w), np.float32)
    prev = None
    for k in range(n):
        fm = focus_measure(np.asarray(stack[k], np.float32), sigma)
        total += fm
        if prev is not None:
            just_after = bestk == k - 1      # the frame after the current best
            after[just_after] = fm[just_after]
        new = fm > best
        best[new] = fm[new]
        bestk[new] = k
        before[new] = prev[new] if prev is not None else fm[new]
        after[new] = fm[new]                 # placeholder until frame k+1 arrives
        prev = fm
    denom = before - 2 * best + after
    off = np.where(denom < -1e-12, 0.5 * (before - after) / np.where(denom < -1e-12, denom, -1), 0.0)
    depth = bestk + np.clip(off, -0.5, 0.5)
    mean = total / n
    conf = np.clip((best - mean) / (best + 1e-12), 0.0, 1.0)
    # texture strength: weak peaks in flat regions are unreliable
    strength = best / (np.percentile(best, 90) + 1e-12)
    conf = conf * np.clip(strength, 0.0, 1.0)
    return depth.astype(np.float32), conf.astype(np.float32)


def smooth_depth(depth, conf, sigma: float = 6.0):
    """Confidence-weighted (normalised) smoothing: fills textureless areas."""
    num = cv2.GaussianBlur(depth * conf, (0, 0), sigma)
    den = cv2.GaussianBlur(conf, (0, 0), sigma)
    glob = float(np.sum(depth * conf) / (np.sum(conf) + 1e-12))
    return np.where(den > 1e-4, num / np.maximum(den, 1e-12), glob).astype(np.float32)
