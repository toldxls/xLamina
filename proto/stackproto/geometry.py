"""Output geometry: crop policies, coverage, and the um/px a stacked image really has.

Input: per-frame transforms (z_k, t_k) relative to the chosen REFERENCE frame
(T_k maps reference-grid centred px -> frame-k centred px; see sim.py), full
resolution.

Policies
--------
native-crop  reference grid, cropped to the area EVERY frame covers. um/px is the
             calibration exactly (at the reference focal depth).
native-full  reference grid, full sensor size; borders are filled from whichever
             frames cover them (coverage map says how many).
rescaled     native-crop, resampled back to the input WIDTH (aspect kept). Pixel
             count matches the input; um/px changes by the reported factor.

Depth caveat: a feature sharp in frame k is drawn in the output at s_k times its
reference-depth size, so effective um/px at that depth = um/px_cal * s_k.
"""
from __future__ import annotations

import math

import numpy as np

from . import sim

POLICIES = ("native-crop", "native-full", "rescaled")


def frame_footprints(z, t, W, H):
    """Corners of every frame's valid area in reference coords (plain px). (N, 4) complex."""
    c = sim.centre(W, H)
    corners = np.array([0, W - 1, (W - 1) + 1j * (H - 1), 1j * (H - 1)]) - c
    iz, it = sim.invert(z, t)
    return iz[:, None] * corners[None, :] + it[:, None] + c


def common_rect(z, t, W, H, margin=2.0):
    """Largest axis-aligned integer pixel rect inside every frame's footprint.

    Conservative for the tiny rotations of a focus stack. ``margin`` keeps the
    interpolation kernel inside real data (2 px suits bicubic). Returns
    (x0, y0, x1, y1) inclusive pixel indices.
    """
    fp = frame_footprints(z, t, W, H)
    # corner order: TL, TR, BR, BL
    left = np.max(np.maximum(fp[:, 0].real, fp[:, 3].real))
    right = np.min(np.minimum(fp[:, 1].real, fp[:, 2].real))
    top = np.max(np.maximum(fp[:, 0].imag, fp[:, 1].imag))
    bottom = np.min(np.minimum(fp[:, 2].imag, fp[:, 3].imag))
    x0 = max(0, math.ceil(left + margin))
    y0 = max(0, math.ceil(top + margin))
    x1 = min(W - 1, math.floor(right - margin))
    y1 = min(H - 1, math.floor(bottom - margin))
    return x0, y0, x1, y1


def coverage_map(z, t, W, H, down=8, margin=2.0):
    """How many frames cover each reference pixel, on a 1/``down`` grid."""
    c = sim.centre(W, H)
    xs = np.arange(0, W, down, dtype=np.float64)
    ys = np.arange(0, H, down, dtype=np.float64)
    P = (xs[None, :] + 1j * ys[:, None]) - c
    cov = np.zeros(P.shape, np.int32)
    for zk, tk in zip(z, t):
        q = zk * P + tk + c
        cov += ((q.real >= margin) & (q.real <= W - 1 - margin)
                & (q.imag >= margin) & (q.imag <= H - 1 - margin))
    return cov


def choose_reference(z0, t0, rule: str | int) -> int:
    """Pick the reference index. z0/t0 are relative to frame 0."""
    n = len(z0)
    if isinstance(rule, int) or str(rule).lstrip("-").isdigit():
        return int(rule) % n
    # T maps ref->k, so s_k > 1 means a reference distance d spans s_k*d px in
    # frame k: frame k's content is LARGER (more zoomed in) than frame 0's.
    # Most-zoomed reference => every other frame is enlarged => no scale crop.
    s = np.abs(z0)
    return {"first": 0, "last": n - 1, "middle": n // 2,
            "most-zoomed": int(np.argmax(s)),
            "least-zoomed": int(np.argmin(s))}[rule]


def report(z, t, W, H, um_per_px=None, ref_index=0, ref_rule="middle",
           selected="native-crop", margin=2.0, coverage_down=8):
    """Geometry report dict for transforms relative to the reference."""
    n = len(z)
    s = np.abs(z)
    x0, y0, x1, y1 = common_rect(z, t, W, H, margin)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    cov = coverage_map(z, t, W, H, coverage_down, margin)
    full_frac = float(np.mean(cov == n))
    # depth: content sharp in frame k is rendered at s_k x its reference-depth size
    # (T maps ref->k, so a ref-grid distance d samples frame k over s_k*d px of a
    # feature imaged at the calibrated magnification -> um per output px = cal*s_k)
    srange = (float(s.min()), float(s.max()))

    def scale_block(upp_factor):
        if um_per_px is None:
            return {"um_per_px": None, "note": "no calibration given (use --fov-mm or --um-per-px)"}
        u = um_per_px * upp_factor
        return {"um_per_px": u,
                "um_per_px_range_over_depth": [u * srange[0], u * srange[1]],
                "range_pct": [100 * (srange[0] - 1), 100 * (srange[1] - 1)]}

    pol = {
        "native-crop": {
            "width": cw, "height": ch,
            "crop_px": {"left": x0, "right": W - 1 - x1, "top": y0, "bottom": H - 1 - y1},
            "pixels_kept_pct": 100.0 * cw * ch / (W * H),
            **scale_block(1.0),
        },
        "native-full": {
            "width": W, "height": H,
            "crop_px": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "full_coverage_pct": 100.0 * full_frac,
            "min_frames_covering": int(cov.min()),
            **scale_block(1.0),
        },
    }
    f = W / cw
    pol["rescaled"] = {
        "width": W, "height": int(round(ch * f)),
        "resample_factor": f,
        "crop_px": pol["native-crop"]["crop_px"],
        **scale_block(1.0 / f),
    }
    if um_per_px is not None:
        for p in pol.values():
            p["fov_mm"] = [p["width"] * p["um_per_px"] / 1000.0, p["height"] * p["um_per_px"] / 1000.0]

    half_diag = math.hypot(W, H) / 2.0
    eps_total = srange[1] / srange[0] - 1.0
    return {
        "input": {"width": W, "height": H, "frames": n,
                  "um_per_px_calibrated": um_per_px},
        "reference": {"index": int(ref_index), "rule": ref_rule},
        "selected_policy": selected,
        "policies": pol,
        "scale_spread": {
            "s_min": srange[0], "s_max": srange[1],
            "total_pct": 100.0 * eps_total,
            "corner_shift_px": eps_total * half_diag,
        },
        "_coverage": cov,            # array; stripped before JSON
        "_rect": (x0, y0, x1, y1),
    }
