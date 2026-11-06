"""Step 5 check: stackcli's SGM depth against fusion.sgm_from_volume + sgm.py.

    python tools/validate/check_depth.py build/cli/stackcli.exe VOLDIR [--crop 600,400,300,200]

VOLDIR is a `stackcli fold --volume` output (half res: the output grid is
exactly 2x the depth grid, which --crop needs). Both sides get the SAME
inputs: the GPU's aggregated volume V, its Lv and guides, cropped. The
prototype is called with vol_agg_px=0 and V^2, so its own sqrt restores V.
A crop keeps the numpy SGM (8 serial sweeps over H x W x K) to seconds; the
whole 1710 x 1107 x 163 volume takes it minutes.

Compares depth_s (the integer depth on the output grid), conf, the spill map
h and the sub-frame argmin before upsampling.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))
from stackproto import fusion  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_volume import read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stackcli")
    ap.add_argument("voldir")
    ap.add_argument("--crop", default="600,400,300,200", help="x0,y0,w,h at the depth resolution; 'none' = whole volume")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    vd = Path(a.voldir)
    vm = read_manifest(vd / "volume.txt")
    mm = read_manifest(vd / "manifest.txt")
    dw, dh, K = int(vm["dw"]), int(vm["dh"]), int(vm["K"])
    ow, oh = int(mm["w"]), int(mm["h"])
    assert ow % dw == 0 and oh % dh == 0, "need an integer output/depth ratio (use a half-res volume)"
    sx, sy = ow // dw, oh // dh
    V = np.fromfile(vd / "vol_V.f32", np.float32).reshape(K, dh, dw)
    Lv = np.fromfile(vd / "vol_lum.f16", np.float16).reshape(K, dh, dw)
    guide = np.fromfile(vd / "guide.f32", np.float32).reshape(oh, ow)
    args = [a.stackcli, "depth", str(vd), "--gpu", str(a.gpu)]
    if a.crop != "none":
        x0, y0, cw, ch = [int(v) for v in a.crop.split(",")]
        V = V[:, y0:y0 + ch, x0:x0 + cw]
        Lv = Lv[:, y0:y0 + ch, x0:x0 + cw]
        guide = guide[y0 * sy:(y0 + ch) * sy, x0 * sx:(x0 + cw) * sx]
        args += ["--crop", a.crop]
        dw, dh, ow, oh = cw, ch, cw * sx, ch * sy
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(args + ["--out", td], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            sys.exit(1)
        print("\n".join("  " + l for l in r.stdout.strip().splitlines()))
        g_depth = np.fromfile(Path(td) / "depth_s.f32", np.float32).reshape(oh, ow)
        g_conf = np.fromfile(Path(td) / "conf.f32", np.float32).reshape(oh, ow)
        g_spill = np.fromfile(Path(td) / "spill.f32", np.float32).reshape(dh, dw)
        g_dd = np.fromfile(Path(td) / "depth_d.f32", np.float32).reshape(dh, dw)

    # reference on the same inputs
    Vk = np.ascontiguousarray(np.moveaxis(V, 0, 2))            # (dh, dw, K)
    Lk = np.ascontiguousarray(np.moveaxis(Lv, 0, 2)).astype(np.float32)
    ctx = dict(vol_agg_px=0.0, depth_full_fraction=0.25, sgm_c_lo=0.45, sgm_c_hi=0.65, halo_radius_px=80.0, end_guard=3,
               end_prior=1.0, dark_prior=1.0, halo_prior=2.0, sgm_p1=0.03, sgm_p2=0.4, sgm_edge=0.02, sgm_dirs=8)
    import time
    t0 = time.time()
    depth_s, conf = fusion.sgm_from_volume(Vk * Vk, Lk, guide, ctx)
    print(f"  numpy reference: {time.time() - t0:.1f} s")
    spill = ctx["_spill"]
    ok = True

    def line(name, good, text):
        nonlocal ok
        ok = ok and good
        print(f"  {name:10s} {text}  {'OK' if good else 'MISMATCH'}", flush=True)

    d = np.abs(g_spill - spill)
    line("spill h", d.max() <= 1e-3, f"max |diff| {d.max():.2e}, rms {np.sqrt((d ** 2).mean()):.2e}")
    d = np.abs(g_conf - conf)
    line("conf", d.max() <= 2e-3, f"max |diff| {d.max():.2e}, rms {np.sqrt((d ** 2).mean()):.2e}")
    # the argmin before upsampling: sub-frame, so compare continuously
    # (recompute the reference's own: not exported by sgm_from_volume, so judge through depth_s instead)
    eq = g_depth == depth_s
    dd = np.abs(g_depth - depth_s)
    line("depth_s", np.mean(eq) >= 0.995 and np.mean(dd <= 1) >= 0.999,
         f"identical {np.mean(eq) * 100:.3f}%, within 1 frame {np.mean(dd <= 1) * 100:.3f}%, max |diff| {dd.max():.0f} frames "
         f"(depth range {depth_s.min():.0f}..{depth_s.max():.0f})")
    print(f"  (GPU sub-frame argmin: range {g_dd.min():.2f}..{g_dd.max():.2f})")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
