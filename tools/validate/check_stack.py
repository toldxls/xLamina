"""End-to-end check: `stackcli stack` outputs against the prototype's fusion of the same stack.

    python tools/validate/check_stack.py <stack out dir> <python fusion dir, e.g. $STACKER_OUT/fusion_0.5>
        [--score]      also score the GPU's B against Helicon B like tools/bench/score_b.py
                       (needs $STACKER_OUT/helicon_on_grid.pkl and helicon_B_depth_quarter.npy)

Per output (A, B, B0, C, CB, Cn, CBn, Cd, Cg, Cg2, H): max and rms difference on
the [0, 1] scale, and the share of pixels within 1 and 4 LSB of 16 bits. The
depth map: share of identical pixels and within one frame.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))
sys.path.insert(0, str(REPO / "tools" / "bench"))
from stackproto.frames import luma, read_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu_dir")
    ap.add_argument("py_dir")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    g, p = Path(a.gpu_dir), Path(a.py_dir)
    names = ["A", "B", "B0", "C", "CB", "Cn", "CBn", "Cd", "Cg", "Cg2", "H"]
    lsb = 1 / 65535
    print(f"{'output':6s} {'max':>9s} {'rms':>9s} {'<=1 LSB':>8s} {'<=4 LSB':>8s}")
    for n in names:
        if not (g / f"{n}.tif").exists() or not (p / f"{n}.tif").exists():
            print(f"{n:6s} missing")
            continue
        x = read_rgb(g / f"{n}.tif")
        y = read_rgb(p / f"{n}.tif")
        d = np.abs(x - y)
        print(f"{n:6s} {d.max():9.2e} {np.sqrt((d ** 2).mean()):9.2e} {100 * np.mean(d <= 1.5 * lsb):7.2f}% {100 * np.mean(d <= 4.5 * lsb):7.2f}%", flush=True)
    dp = p / "depth.npy"
    dg = g / "depth.f32"
    if dp.exists() and dg.exists():
        D_py = np.load(dp).astype(np.float32)
        D_g = np.fromfile(dg, np.float32).reshape(D_py.shape)
        dd = np.abs(D_g - D_py)
        print(f"depth  identical {100 * np.mean(dd == 0):.3f}%, within 1 frame {100 * np.mean(dd <= 1):.3f}%, within 3 {100 * np.mean(dd <= 3):.3f}%, "
              f"max {dd.max():.0f} frames")
    if a.score:
        out = Path(os.environ.get("STACKER_OUT", str(REPO / "out" / "tiff_2026-11-05")))
        hel = pickle.load(open(out / "helicon_on_grid.pkl", "rb"))
        hb = hel["Helicon B"]
        valid = cv2.erode((hb.sum(2) > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        yh = luma(hb)
        gb = cv2.GaussianBlur(yh, (0, 0), 1.5)
        grad = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
        edge = grad > np.percentile(grad[valid], 95)
        band = ((cv2.dilate(edge.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
                & ~(cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & valid)
        hdp = out / "helicon_B_depth_quarter.npy"
        hd = np.load(hdp).astype(np.float32) if hdp.exists() else None

        def tone_match(img):
            o = np.empty_like(img)
            for c in range(3):
                qs = np.linspace(0, 1, 1001)
                o[:, :, c] = np.interp(img[:, :, c], np.quantile(img[:, :, c][valid], qs), np.quantile(hb[:, :, c][valid], qs))
            return o

        for label, B, depth in (("GPU B", read_rgb(g / "B.tif"), np.fromfile(dg, np.float32).reshape(hb.shape[:2]) if dg.exists() else None),
                                ("Python B", read_rgb(p / "B.tif"), np.load(dp).astype(np.float32) if dp.exists() else None)):
            m = tone_match(B)
            e = (m - hb) ** 2
            d = cv2.GaussianBlur(luma(m) - yh, (0, 0), 3)
            line = "%-9s RMSE all %.4f  edge-band %.4f  edge glare %.1f%%" % (label, np.sqrt(e[valid].mean()), np.sqrt(e[band].mean()),
                                                                               100 * np.mean(d[band] > 0.03))
            if hd is not None and depth is not None:
                dq = cv2.resize(depth, (hd.shape[1], hd.shape[0]), interpolation=cv2.INTER_NEAREST)
                vq = cv2.resize(valid.astype(np.uint8), (hd.shape[1], hd.shape[0])) > 0
                line += "  depth~Helicon(<=5fr) %.0f%%" % (100 * np.mean(np.abs(dq - hd)[vq] <= 5))
            print(line, flush=True)


if __name__ == "__main__":
    main()
