"""Measure the geometry of an EXTERNAL stacker's output (e.g. Helicon) against
our alignment of the same frames.

1. Quick all-in-focus luma at work resolution on the reference grid
   (focus-measure argmax over the aligned frames).
2. Register the external output (similarity) to it.
3. Report the output's scale relative to the reference frame, which source frame
   that scale matches (= the external tool's reference), crop per edge in
   reference pixels, and the effective um/px.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import align, sim
from .depth import focus_measure
from .frames import luma, read_rgb


def quick_fusion(stack, zr, tr_work):
    """Argmax-focus fusion of a luma stack onto the reference grid (work res)."""
    n, h, w = stack.shape
    best = np.full((h, w), -1.0, np.float32)
    fused = np.zeros((h, w), np.float32)
    idx = np.zeros((h, w), np.int16)
    for k in range(n):
        M = sim.to_affine(zr[k], tr_work[k], w, h)
        im = cv2.warpAffine(np.asarray(stack[k], np.float32), M, (w, h),
                            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        fm = focus_measure(im, 3.0)
        m = fm > best
        best[m] = fm[m]
        fused[m] = im[m]
        idx[m] = k
    return fused, idx


def register_external(ext_path, stack, report, factors):
    fx, fy = factors
    W, H = report["geometry"]["input"]["width"], report["geometry"]["input"]["height"]
    pf = report["per_frame"]
    zr = np.array([p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"])) for p in pf])
    tr_full = np.array([complex(p["tx"], p["ty"]) for p in pf])
    tr_work = tr_full.real * fx + 1j * tr_full.imag * fy
    fused, idx = quick_fusion(stack, zr, tr_work)
    h, w = fused.shape

    ext = luma(read_rgb(ext_path))
    EH, EW = ext.shape
    ew, eh = int(round(EW * fx)), int(round(EH * fy))
    ext_w = cv2.resize(ext, (ew, eh), interpolation=cv2.INTER_AREA)
    efx, efy = ew / EW, eh / EH

    # coarse offset: pad the external image to our size, phase-correlate
    pad = np.full((h, w), float(ext_w.mean()), np.float32)
    pad[:eh, :ew] = ext_w
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(fused, pad, win)
    ox, oy = int(round(-dx)), int(round(-dy))
    ox, oy = int(np.clip(ox, 0, w - ew)), int(np.clip(oy, 0, h - eh))
    crop = np.ascontiguousarray(fused[oy:oy + eh, ox:ox + ew])

    pa, f = align.pyramid(crop, 150)
    pb, _ = align.pyramid(ext_w, 150)
    r = align.align_pair(pa, pb, f)
    # P maps crop-centred work coords -> external-centred work coords.
    # Map the external image's corners back into reference FULL-res pixels.
    cc = sim.centre(ew, eh)
    ce = sim.centre(ew, eh)
    corners_e = np.array([0, EW - 1, (EW - 1) + 1j * (EH - 1), 1j * (EH - 1)])
    q = (corners_e.real * efx + 1j * corners_e.imag * efy) - ce          # ext centred work
    p_crop = (q - r.t) / r.z                                              # crop centred work
    p_ref_work = p_crop + cc + complex(ox, oy)                            # ref plain work px
    p_ref = p_ref_work.real / fx + 1j * p_ref_work.imag / fy              # ref full px (approx)
    s_ext = abs(r.z) * (efx / fx)                    # ext px per ref px (full res)
    s_frames = np.array([p["s"] for p in pf])
    k_star = int(np.argmin(np.abs(s_frames - s_ext)))
    upp = report["geometry"]["input"]["um_per_px_calibrated"]
    left = float(np.mean([p_ref[0].real, p_ref[3].real]))
    right = float(W - 1 - np.mean([p_ref[1].real, p_ref[2].real]))
    top = float(np.mean([p_ref[0].imag, p_ref[1].imag]))
    bottom = float(H - 1 - np.mean([p_ref[2].imag, p_ref[3].imag]))
    out = {
        "external": Path(ext_path).name,
        "external_size": [EW, EH],
        "reference_frame": report["geometry"]["reference"]["index"],
        "scale_vs_reference": s_ext,
        "scale_vs_reference_ppm": 1e6 * (s_ext - 1),
        "rotation_vs_reference_mdeg": 1e3 * float(np.degrees(np.angle(r.z))),
        "matches_frame_scale": {"index": k_star, "file": pf[k_star]["file"],
                                "frame_scale": float(s_frames[k_star])},
        "footprint_in_reference_px": {"left": left, "right": right, "top": top, "bottom": bottom},
        "registration_ncc": r.ncc,
    }
    if upp:
        # content sharp in frame k sits at upp*s_k on the reference grid, and the
        # external grid is the reference scaled by s_ext
        out["um_per_px_external"] = upp / s_ext
        out["um_per_px_external_over_depth"] = {
            "first_frame": upp * float(s_frames[0]) / s_ext,
            "last_frame": upp * float(s_frames[-1]) / s_ext,
            "min": upp * float(s_frames.min()) / s_ext, "max": upp * float(s_frames.max()) / s_ext}
        out["fov_mm_external"] = [EW * upp / s_ext / 1000, EH * upp / s_ext / 1000]
    return out, fused, idx


def warp_external_onto(ext_path, target_rgb, ext_scale):
    """Warp an external output (full-res file) onto ``target_rgb``'s grid.

    ``ext_scale``: factor taking the external file to the target's resolution
    (e.g. 0.5). Registration: similarity on luma (coarse offset by phase
    correlation, then the exact Gauss-Newton pair aligner).
    Returns (warped_rgb, info).
    """
    ext = ext_path if isinstance(ext_path, np.ndarray) else read_rgb(ext_path)
    EH, EW = ext.shape[:2]
    ew, eh = int(round(EW * ext_scale)), int(round(EH * ext_scale))
    ext_s = ext if ext_scale == 1.0 else cv2.resize(ext, (ew, eh), interpolation=cv2.INTER_AREA)
    tgt_y = luma(target_rgb)
    ext_y = luma(ext_s)
    h, w = tgt_y.shape
    # common size for the coarse search
    ch, cw = max(h, eh), max(w, ew)
    A = np.full((ch, cw), float(tgt_y.mean()), np.float32); A[:h, :w] = tgt_y
    B = np.full((ch, cw), float(ext_y.mean()), np.float32); B[:eh, :ew] = ext_y
    win = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(A, B, win)
    ox, oy = int(round(-dx)), int(round(-dy))
    # overlapping window: target [oy:oy+hh, ox:ox+ww] <-> ext [0:hh, 0:ww] (shifted if negative)
    ex0, ey0 = max(0, -ox), max(0, -oy)
    tx0, ty0 = max(0, ox), max(0, oy)
    ww = min(w - tx0, ew - ex0)
    hh = min(h - ty0, eh - ey0)
    tc = np.ascontiguousarray(tgt_y[ty0:ty0 + hh, tx0:tx0 + ww])
    ec = np.ascontiguousarray(ext_y[ey0:ey0 + hh, ex0:ex0 + ww])
    pa, f = align.pyramid(tc, 150)
    pb, _ = align.pyramid(ec, 150)
    r = align.align_pair(pa, pb, f)
    # target plain (x) -> window-centred p -> ext-window-centred q -> ext plain
    c = sim.centre(ww, hh)
    M = sim.to_affine(r.z, r.t, ww, hh)              # window plain -> ext-window plain
    M[:, 2] += M[:, :2] @ np.array([-tx0, -ty0]) + np.array([ex0, ey0])
    warped = cv2.warpAffine(ext_s, M, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped, {"scale": abs(r.z), "rot_mdeg": 1e3 * float(np.degrees(np.angle(r.z))),
                    "ncc": r.ncc, "offset": (ox, oy)}
