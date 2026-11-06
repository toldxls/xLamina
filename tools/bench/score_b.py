"""Render B variants on the half-res grid and score them against Helicon's B
(tone-matched RMSE overall / in the edge band, edge-band glare %, agreement with
Helicon's recovered depth), plus a crop montage on the Desktop.

    python tools/bench/score_b.py '{"default": {}, "agg16": {"vol_agg_px": 16}}'
"""
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from common import FRAMES, FUSION, HALF_RES_ROIS, OUT

if __name__ == "__main__":
    from stackproto import montage
    from stackproto.depthlab import Lab
    from stackproto.frames import luma, write_rgb16
    variants = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"default": {}}
    hel = pickle.load(open(OUT / "helicon_on_grid.pkl", "rb"))
    hb = hel["Helicon B"]
    valid = cv2.erode((hb.sum(2) > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    yh = luma(hb)
    g = cv2.GaussianBlur(yh, (0, 0), 1.5)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edge = grad > np.percentile(grad[valid], 95)
    band = ((cv2.dilate(edge.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
            & ~(cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & valid)
    hdp = OUT / "helicon_B_depth_quarter.npy"
    hd = np.load(hdp).astype(np.float32) if hdp.exists() else None

    def tone_match(img):
        out = np.empty_like(img)
        for c in range(3):
            qs = np.linspace(0, 1, 1001)
            out[:, :, c] = np.interp(img[:, :, c], np.quantile(img[:, :, c][valid], qs),
                                     np.quantile(hb[:, :, c][valid], qs))
        return out
    L = Lab(OUT, FRAMES)
    names = []
    for name, kw in variants.items():
        out, info = L.fuse(log=lambda *_: None, **kw)
        write_rgb16(FUSION / f"B_{name}.tif", out["B"])
        m = tone_match(out["B"])
        e = (m - hb) ** 2
        d = cv2.GaussianBlur(luma(m) - yh, (0, 0), 3)
        line = "%-16s RMSE all %.4f  edge-band %.4f  edge glare %.1f%%" % (
            name, np.sqrt(e[valid].mean()), np.sqrt(e[band].mean()), 100 * np.mean(d[band] > 0.03))
        if hd is not None:
            dq = cv2.resize(info["depth"], (hd.shape[1], hd.shape[0]), interpolation=cv2.INTER_NEAREST)
            vq = cv2.resize(valid.astype(np.uint8), (hd.shape[1], hd.shape[0])) > 0
            line += "  depth~Helicon(<=5fr) %.0f%%" % (100 * np.mean(np.abs(dq - hd)[vq] <= 5))
        print(line, flush=True)
        names.append(f"B_{name}")
    montage.build(str(FUSION), [("Helicon B", None)], names,
                  str(Path.home() / "Desktop" / "stacker_B_crops.jpg"), rois=HALF_RES_ROIS, cache=hel)
