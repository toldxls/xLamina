"""Score fusion methods against a synthetic stack's ground truth.

Regions: 'halo zone' = within 12 px of a depth discontinuity, 'plateaus' = the
rest. 'added noise' = RMS of the high-pass of (fused - truth) on plateaus.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import fusion, sim
from .frames import list_frames, read_rgb, write_rgb16
from .synth import load_truth


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    return float(10 * np.log10(1.0 / max(float(np.mean(d)), 1e-12)))


def evaluate(folder, out_dir, workers=None, scale=1.0, **kw):
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, z, t = load_truth(folder)
    z, t = sim.rebase(z, t, 0)
    truth = read_rgb(folder / "_truth_sharp.tif")
    depth = np.load(folder / "_truth_depth.npy")
    H, W = truth.shape[:2]
    m = 8
    grid = fusion.Grid(W, H, (m, m, W - 1 - m, H - 1 - m), scale=scale)
    out, info = fusion.run(list_frames(folder), z, t, grid, workers=workers, **kw)
    tr = truth[m:H - m, m:W - m]
    dp = depth[m:H - m, m:W - m]
    gy, gx = np.gradient(dp)
    edge = (np.hypot(gx, gy) > 2.0 * float(np.median(np.abs(np.diff(dp, axis=1)))) + 1.0)
    halo = cv2.dilate(edge.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    plate = ~halo
    rows = {}
    for k, img in out.items():
        e = img - tr
        hp = e - cv2.GaussianBlur(e, (0, 0), 2.0)
        rows[k] = {"psnr_all": psnr(img, tr), "psnr_halo_zone": psnr(img, tr, halo),
                   "psnr_plateaus": psnr(img, tr, plate),
                   "added_noise_rms": float(np.sqrt(np.mean(hp[plate] ** 2)))}
        write_rgb16(out_dir / f"{k}.tif", img)
    # depth accuracy (frames) vs truth
    dt = dp / d["params"]["step_um"]
    dm = cv2.resize(info["depth"], (dt.shape[1], dt.shape[0]), interpolation=cv2.INTER_NEAREST) \
        if info["depth"].shape != dt.shape else info["depth"]
    err = np.abs(dm - dt)
    rows["_depth"] = {"median_abs_err_frames": float(np.median(err)),
                      "within_1_frame_pct": float(100 * np.mean(err <= 1.0)),
                      "within_1_frame_halo_pct": float(100 * np.mean(err[halo] <= 1.0)),
                      "within_1_frame_plateau_pct": float(100 * np.mean(err[plate] <= 1.0))}
    rows["_replaced_fraction_per_level"] = info["replaced"]
    (out_dir / "scores.json").write_text(json.dumps(rows, indent=1))
    return rows, out, tr, halo
