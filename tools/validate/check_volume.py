"""Step 4 check: stackcli's depth-map input against the prototype.

    python tools/validate/check_volume.py build/cli/stackcli.exe Q:\\TIFF [--frames 0,54,81,108,162] [--scale 0.5]

Runs `stackcli fold --volume` on the chosen frames and compares, on the CPU
for the same frames:
  guide      luma of collapse(c_coef + [c_base / n]) of fusion.fold's state, clipped
  fm, lum    _w_fmvol: INTER_AREA to the depth res, focus_measure_rgb(normalized)
             and blurred luma, as float16
  V          sgm_from_volume's aggregation: guided_filter(guide_d, fm_k, r, 1e-4),
             clamp, sqrt, per frame
Tolerances follow the inputs: fm/lum are float16 (1 ulp = 1e-3 relative), the
guide inherits the warp's 1e-4, V the aggregation of float16 slices.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))
from stackproto import fusion  # noqa: E402
from stackproto.fusion import LUMA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fold import load_state, rel, rms, run_fold, write_list  # noqa: E402


def read_manifest(p):
    m = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            k, v = line.split(" ", 1)
            m[k] = v
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stackcli")
    ap.add_argument("frames_dir")
    ap.add_argument("--report", default=str(REPO / "tests" / "data" / "report_2026-11-05.json"))
    ap.add_argument("--frames", default="0,54,81,108,162")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    rep = json.loads(Path(a.report).read_text())
    pf = rep["per_frame"]
    W, H = rep["geometry"]["input"]["width"], rep["geometry"]["input"]["height"]
    cp = rep["geometry"]["policies"]["native-crop"]["crop_px"]
    rect = (cp["left"], cp["top"], W - 1 - cp["right"], H - 1 - cp["bottom"])
    grid = fusion.Grid(W, H, rect, a.scale)
    ow, oh = grid.size
    levels = fusion.n_levels(ow, oh)
    radius = max(0.8, 6.0 * a.scale)
    frames = [int(v) for v in a.frames.split(",")]
    dff = 0.25
    ds = dff / a.scale
    dw, dh = max(16, int(round(ow * ds))), max(16, int(round(oh * ds)))
    fm_radius = max(0.8, 6.0 * dff)
    agg_r = int(round(32.0 * dff))
    ok = True

    def line(name, good, text):
        nonlocal ok
        ok = ok and good
        print(f"  {name:12s} {text}  {'OK' if good else 'MISMATCH'}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst = td / "list.txt"
        write_list(lst, rep, a.frames_dir)
        print(f"== volume check: frames {frames} at scale {a.scale:g}; depth res {dw}x{dh}, fm sigma {fm_radius}, agg r {agg_r}")
        out = run_fold(a.stackcli, lst, rect, a.scale, td / "gpu", frames=frames, extra=["--volume"])
        for l in out.splitlines():
            if "frames in" in l or "volume:" in l:
                print("  " + l.strip())
        man = read_manifest(td / "gpu" / "volume.txt")
        assert int(man["dw"]) == dw and int(man["dh"]) == dh, (man, dw, dh)
        K = int(man["K"])
        g_guide = np.fromfile(td / "gpu" / "guide.f32", np.float32).reshape(oh, ow)
        g_ga = np.fromfile(td / "gpu" / "guide_d.f32", np.float32).reshape(dh, dw)
        g_fm = np.fromfile(td / "gpu" / "vol_fm.f16", np.float16).reshape(K, dh, dw)
        g_lum = np.fromfile(td / "gpu" / "vol_lum.f16", np.float16).reshape(K, dh, dw)
        g_V = np.fromfile(td / "gpu" / "vol_V.f32", np.float32).reshape(K, dh, dw)

        # reference, same frames on the CPU
        st = fusion.new_state(oh, ow, levels)
        fmv, lumv = {}, {}
        for k in frames:
            p = pf[k]
            z = p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"]))
            img = fusion.load_aligned(Path(a.frames_dir) / p["file"], z, complex(p["tx"], p["ty"]), grid)
            fusion.fold(st, k, img, levels, radius, 4.0)
            imd = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            fmv[k] = fusion.focus_measure_rgb(imd, fm_radius, normalized=True).astype(np.float16)
            lumv[k] = cv2.GaussianBlur(imd @ LUMA, (0, 0), 2.0 * fm_radius).astype(np.float16)
        g_img = fusion.collapse(st.c_coef + [(st.c_base / max(st.count, 1)).astype(np.float32)])
        guide = np.clip(g_img @ LUMA, 0, 1).astype(np.float32)
        ga = cv2.resize(guide, (dw, dh), interpolation=cv2.INTER_AREA)

        # the collapse itself: Python's collapse of the GPU's OWN merged state (the state was checked in step 3;
        # where a C near-tie went the other way the collapsed value legitimately differs)
        gs = load_state(td / "gpu")
        coll_g = fusion.collapse(gs["c_coef"] + [(gs["c_base"] / max(gs["count"], 1)).astype(np.float32)])
        guide_g = np.clip(coll_g @ LUMA, 0, 1).astype(np.float32)
        d = np.abs(g_guide - guide_g)
        line("collapse", d.max() <= 1e-5, f"GPU collapse vs cv2 collapse of the same state: max |diff| {d.max():.2e}, rms {rms(g_guide, guide_g):.2e}")
        d = np.abs(g_guide - guide)
        print(f"  {'(guide e2e)':12s} vs the Python fold's guide: rms {rms(g_guide, guide):.2e}, pixels > 1e-3: {np.mean(d > 1e-3) * 100:.4f}%")
        ga_g = cv2.resize(g_guide, (dw, dh), interpolation=cv2.INTER_AREA)
        d = np.abs(g_ga - ga_g)
        line("area resize", d.max() <= 1e-6, f"GPU INTER_AREA vs cv2 on the same guide: max |diff| {d.max():.2e}")
        for k in frames:
            rf = rel(g_fm[k].astype(np.float32), fmv[k].astype(np.float32), floor=1e-3)
            rl = rel(g_lum[k].astype(np.float32), lumv[k].astype(np.float32), floor=1e-3)
            same_fm = np.mean(g_fm[k] == fmv[k]) * 100
            line(f"fm[{k}]", rf <= 4e-3, f"max rel {rf:.2e} (identical f16 {same_fm:.1f}%), rms {rms(g_fm[k], fmv[k]):.2e}")
            line(f"lum[{k}]", rl <= 2e-3, f"max rel {rl:.2e}")
            # the aggregation itself: Python's guided filter on the GPU's own slice and guide
            Vg = np.sqrt(np.maximum(fusion.guided_filter(g_ga, g_fm[k].astype(np.float32), agg_r, 1e-4), 0.0))
            allowed = 1e-4 + 1e-4 * Vg          # box-filter rounding on sums of ~1e3 values, plus relative slack
            bad = np.abs(g_V[k] - Vg) > allowed
            line(f"V[{k}]", not bad.any(), f"GPU aggregation vs cv2 on the same inputs: max |diff| {np.max(np.abs(g_V[k] - Vg)):.2e}, "
                 f"rms {rms(g_V[k], Vg):.2e}, max V {Vg.max():.3g}")
            Vr = np.sqrt(np.maximum(fusion.guided_filter(ga, fmv[k].astype(np.float32), agg_r, 1e-4), 0.0))
            d = np.abs(g_V[k] - Vr)
            print(f"  {'(V e2e)':12s} vs all-Python: rms {rms(g_V[k], Vr):.2e}, pixels > 1e-2: {np.mean(d > 1e-2) * 100:.4f}%")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
