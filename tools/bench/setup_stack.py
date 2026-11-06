"""One-time setup on a machine: alignment report, half-res fusion, Helicon outputs
warped onto our grid, Helicon's recovered depth, and the cached depth volume.
Every step is skipped if its output already exists.

    python tools/bench/setup_stack.py
"""
import json
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from common import FOV_MM, FRAMES, FUSION, GLOB, OUT, REPO, STEP_UM, helicon

if __name__ == "__main__":
    py = sys.executable
    if not (OUT / "report.json").exists():
        subprocess.check_call([py, "-m", "stackproto", "analyze", str(FRAMES), "--glob", GLOB,
                               "--step-um", str(STEP_UM), "--fov-mm", str(FOV_MM), "--out", str(OUT),
                               "--png", str(OUT / "diagnostics.png")], cwd=REPO / "proto")
    if not (FUSION / "C.tif").exists():
        subprocess.check_call([py, "-m", "stackproto", "fuse", str(OUT), str(FRAMES), "--scale", "0.5",
                               "--save-pyramids"], cwd=REPO / "proto")
    from stackproto import compare, fusion
    from stackproto.depthlab import Lab
    from stackproto.frames import read_rgb
    hp = OUT / "helicon_on_grid.pkl"
    if not hp.exists():
        ref = read_rgb(FUSION / "C.tif")
        hel = {}
        for lab, kind in (("Helicon B", "B"), ("Helicon C", "C"), ("Your B+C comp", "comp")):
            f = helicon(kind)
            if f:
                hel[lab], info = compare.warp_external_onto(f, ref, 0.5)
                print(lab, f.name, {k: v for k, v in info.items() if k != "offset"})
        pickle.dump(hel, open(hp, "wb"), protocol=4)
    hel = pickle.load(open(hp, "rb"))
    hd = OUT / "helicon_B_depth_quarter.npy"
    if "Helicon B" in hel and not hd.exists():
        # which source frame each region of Helicon's B matches best (coarse; see FINDINGS 14)
        L = Lab(OUT, FRAMES)
        hb = cv2.resize(hel["Helicon B"], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        best = np.full(hb.shape[:2], np.inf, np.float32)
        idx = np.zeros(hb.shape[:2], np.int16)

        def one(k):
            im = cv2.resize(fusion.load_aligned(L.paths[k], L.z[k], L.t[k], L.grid), (hb.shape[1], hb.shape[0]),
                            interpolation=cv2.INTER_AREA)
            return k, cv2.boxFilter(((im - hb) ** 2).sum(2), -1, (7, 7))
        with ThreadPoolExecutor(6) as ex:
            for k, ssd in ex.map(one, range(len(L.paths))):
                m = ssd < best
                best[m] = ssd[m]
                idx[m] = k
        np.save(hd, idx)
    print("setup complete:", OUT)
