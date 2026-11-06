"""Fast depth experiments on the real stack: cached focus/brightness volume ->
cost -> SGM -> depth, compared with Helicon's reverse-engineered depth."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import fusion


class Lab:
    def __init__(self, analysis_dir, frames_dir, scale=0.5, workers=6):
        self.dir = Path(analysis_dir)
        rep = json.loads((self.dir / "report.json").read_text())
        pf = rep["per_frame"]
        W, H = rep["geometry"]["input"]["width"], rep["geometry"]["input"]["height"]
        self.z = np.array([p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"])) for p in pf])
        self.t = np.array([complex(p["tx"], p["ty"]) for p in pf])
        self.paths = [Path(frames_dir) / p["file"] for p in pf]
        cp = rep["geometry"]["policies"]["native-crop"]["crop_px"]
        self.grid = fusion.Grid(W, H, (cp["left"], cp["top"], W - 1 - cp["right"], H - 1 - cp["bottom"]),
                                scale)
        self.workers = workers
        self.cache = str(self.dir / f"fmvol_{scale:g}.npz")

    def fuse(self, **kw):
        kw.setdefault("depth_method", "sgm")
        return fusion.run(self.paths, self.z, self.t, self.grid, workers=self.workers,
                          vol_cache=self.cache, **kw)


def depth_only(lab: Lab, **params):
    """Depth from the cached volume, no frame reads (seconds, not minutes)."""
    from .frames import luma, read_rgb
    zv = np.load(lab.cache)
    V, Lv = zv["fm"].astype(np.float32), zv["lum"].astype(np.float32)
    guide = np.clip(luma(read_rgb(lab.dir / f"fusion_{lab.grid.scale:g}" / "C.tif")), 0, 1)
    ctx = dict(sgm_p1=0.03, sgm_p2=0.4, sgm_edge=0.02, sgm_dirs=8, dark_prior=1.0,
               sgm_c_lo=0.45, sgm_c_hi=0.60)
    ctx.update(params)
    return fusion.sgm_from_volume(V, Lv, guide, ctx)


REGIONS = {"background": np.s_[100:900, 1450:1690], "matrix": np.s_[800:1050, 100:600],
           "crystal fan": np.s_[480:720, 850:1100], "dark gap": np.s_[500:560, 1080:1130]}


def agreement(depth, helicon_q, regions=REGIONS):
    """% of pixels within 5 frames of Helicon's depth, per region (depth-res coords)."""
    hd = cv2.resize(helicon_q.astype(np.float32), (depth.shape[1] // 2, depth.shape[0] // 2),
                    interpolation=cv2.INTER_NEAREST)
    d = cv2.resize(depth, (hd.shape[1], hd.shape[0]), interpolation=cv2.INTER_NEAREST)
    return {n: round(100 * float(np.mean(np.abs(d[sl] - hd[sl]) <= 5)), 0) for n, sl in regions.items()}
