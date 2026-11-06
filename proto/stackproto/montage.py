"""Side-by-side crops: external (Helicon) outputs vs ours, on our fusion grid."""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import compare
from .frames import read_rgb

DEFAULT_ROIS = [(1700, 950, "crystal fan"), (650, 1500, "crystal edge"),
                (2200, 720, "specimen / background edge")]


def build(fusion_dir, externals, methods, out_path, rois=DEFAULT_ROIS, size=440, ext_scale=0.5,
          cache=None):
    ours = {k: read_rgb(os.path.join(fusion_dir, k + ".tif")) for k in methods}
    ref = ours.get("C", next(iter(ours.values())))
    hel = cache if cache is not None else {}
    for lab, f in externals:
        if lab not in hel:
            hel[lab], _ = compare.warp_external_onto(f, ref, ext_scale)
    cols = [(lab, hel[lab]) for lab, _ in externals] + [("ours " + k, ours[k]) for k in methods]
    rows = []
    for x, y, name in rois:
        tiles = []
        for lab, img in cols:
            c = (np.clip(img[y:y + size, x:x + size], 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
            cv2.rectangle(c, (0, 0), (size, 26), (0, 0, 0), -1)
            cv2.putText(c, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(c, ((0, 0), (0, 4), (0, 0)), constant_values=40))
        r = np.hstack(tiles)
        band = np.full((30, r.shape[1], 3), 25, np.uint8)
        cv2.putText(band, f"{name}  (x {x}, y {y} on the {ext_scale:g}x grid)", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([band, r]))
    m = np.vstack(rows)
    cv2.imwrite(out_path, m, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return m, hel


def build_fullres(fusion_dir, externals, methods, out_path, ext_origin, rois, size=380,
                  ext_pad=1040, cache=None):
    """1:1 crops from a full-res ROI fusion. ``ext_origin`` = approximate (x, y) of
    the ROI's corner in the external images (registration refines it)."""
    ours = {k: read_rgb(os.path.join(fusion_dir, k + ".tif")) for k in methods}
    ref = ours.get("C", next(iter(ours.values())))
    hel = cache if cache is not None else {}
    x0, y0 = ext_origin
    for lab, f in externals:
        if lab not in hel:
            big = read_rgb(f)
            crop = np.ascontiguousarray(big[y0:y0 + ext_pad, x0:x0 + ext_pad])
            del big
            hel[lab], _ = compare.warp_external_onto(crop, ref, 1.0)
    cols = [(lab, hel[lab]) for lab, _ in externals] + [("ours " + k, ours[k]) for k in methods]
    rows = []
    for x, y, name in rois:
        tiles = []
        for lab, img in cols:
            c = (np.clip(img[y:y + size, x:x + size], 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
            cv2.rectangle(c, (0, 0), (size, 24), (0, 0, 0), -1)
            cv2.putText(c, lab, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(c, ((0, 0), (0, 4), (0, 0)), constant_values=40))
        r = np.hstack(tiles)
        band = np.full((28, r.shape[1], 3), 25, np.uint8)
        cv2.putText(band, name + " - FULL RES 1:1", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([band, r]))
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return hel


def sharpness_ratio(img, ref, rois, size=440):
    """Mean |Laplacian| of img / ref per ROI (1.0 = as sharp as the reference)."""
    def lap(x):
        y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return np.abs(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 0.7), cv2.CV_32F, ksize=3))
    la, lb = lap(img), lap(ref)
    return {name: round(float(la[y:y + size, x:x + size].mean() / (lb[y:y + size, x:x + size].mean() + 1e-12)), 3)
            for x, y, name in rois}
