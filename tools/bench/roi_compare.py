"""Full-resolution 1:1 comparison of B variants against Helicon B in fixed ROIs
(the regions Travis flagged). Writes ~/Desktop/stacker_B_roi_<name>.jpg.

    python tools/bench/roi_compare.py '{"default": {}, "no agg": {"vol_agg_px": 0}}' plates fan
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from common import FRAMES, FULL_RES_ROIS, OUT, helicon

if __name__ == "__main__":
    from stackproto import compare, fusion
    from stackproto.depthlab import Lab
    from stackproto.frames import read_rgb
    variants = json.loads(sys.argv[1])
    rois = sys.argv[2:] or list(FULL_RES_ROIS)
    L = Lab(OUT, FRAMES)
    W, H = L.grid.W, L.grid.H
    hb_full = read_rgb(helicon("B"))
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    for rname in rois:
        x0, y0 = FULL_RES_ROIS[rname]
        g = fusion.Grid(W, H, (x0, y0, x0 + 799, y0 + 799), 1.0)
        res = {}
        for vname, kw in variants.items():
            o, _ = fusion.run(L.paths, L.z, L.t, g, workers=L.workers, log=lambda *_: None, **kw)
            res[vname] = o["B"]
        # Helicon referenced frame 0 at ~0.989 scale with ~(58, 74) px offset on this stack;
        # take a generous crop and let the registration find it exactly
        ex, ey = int((x0 - 58) * 0.989) - 120, int((y0 - 74) * 0.989) - 120
        crop = np.ascontiguousarray(hb_full[max(0, ey):ey + 1040, max(0, ex):ex + 1040])
        hw, _ = compare.warp_external_onto(crop, next(iter(res.values())), 1.0)
        pickle.dump({"helicon": hw, **res}, open(OUT / f"roi_{rname}.pkl", "wb"))
        tiles = []
        for lab, img in [("Helicon B", hw)] + list(res.items()):
            c = u8(img[100:540, 100:540])
            cv2.rectangle(c, (0, 0), (440, 22), (0, 0, 0), -1)
            cv2.putText(c, lab + "  (1:1)", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.pad(c, ((0, 4), (0, 4), (0, 0)), constant_values=40))
        cv2.imwrite(str(Path.home() / "Desktop" / f"stacker_B_roi_{rname}.jpg"), np.hstack(tiles),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", rname)
