"""Step 2 check: stackcli's GPU warp against fusion.load_aligned.

    python tools/validate/check_warp.py build/cli/stackcli.exe Q:\\TIFF [--report tests/data/report_2026-11-05.json]
                                        [--frames 0,81,162] [--scales 1,0.5]

For each frame and scale, runs `stackcli warp` twice (OpenCV-quantized and
exact positions) with that frame's (z, t) and the report's native-crop grid,
and compares with load_aligned (cv2.resize INTER_AREA + warpAffine
INTER_CUBIC, BORDER_REFLECT). Passes when the exact-position GPU warp (the
product path) is within tolerance of the prototype: rms <= --rms (1e-5) and
max <= --max (1e-3) on the [0, 1] scale; measured 1e-6 / 7e-5 against cv2 5.0.
The OpenCV-4-style quantized mode is reported for the record. Each frame is
read once on the CPU.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))
from stackproto import fusion  # noqa: E402


def run(cli, args):
    r = subprocess.run([cli] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"stackcli failed: {r.stdout}{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stackcli")
    ap.add_argument("frames_dir")
    ap.add_argument("--report", default=str(REPO / "tests" / "data" / "report_2026-11-05.json"))
    ap.add_argument("--frames", default="0,81,162")
    ap.add_argument("--scales", default="1,0.5")
    ap.add_argument("--rms", type=float, default=1e-5)
    ap.add_argument("--max", type=float, default=1e-3)
    a = ap.parse_args()

    rep = json.loads(Path(a.report).read_text())
    pf = rep["per_frame"]
    W, H = rep["geometry"]["input"]["width"], rep["geometry"]["input"]["height"]
    cp = rep["geometry"]["policies"]["native-crop"]["crop_px"]
    rect = (cp["left"], cp["top"], W - 1 - cp["right"], H - 1 - cp["bottom"])
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for scale in [float(s) for s in a.scales.split(",")]:
            grid = fusion.Grid(W, H, rect, scale)
            ow, oh = grid.size
            for k in [int(v) for v in a.frames.split(",")]:
                p = pf[k]
                path = Path(a.frames_dir) / p["file"]
                z = p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"]))
                t = complex(p["tx"], p["ty"])
                ref = fusion.load_aligned(path, z, t, grid)
                assert ref.shape == (oh, ow, 3), ref.shape
                common = ["warp", str(path), None, "--z", f"{z.real:.17g},{z.imag:.17g}", "--t", f"{t.real:.17g},{t.imag:.17g}",
                          "--rect", ",".join(str(v) for v in rect), "--scale", f"{scale:g}"]
                res = {}
                for mode, extra in (("quantized", ["--quantize"]), ("exact", [])):
                    raw = Path(td) / f"{mode}.f32"
                    args = list(common)
                    args[2] = str(raw)
                    out = run(a.stackcli, args + extra)
                    ms = [l for l in out.splitlines() if "load_aligned:" in l]
                    res[mode] = (np.fromfile(raw, np.float32).reshape(ref.shape), ms[0].strip() if ms else "")
                    raw.unlink(missing_ok=True)
                q, e = res["quantized"][0], res["exact"][0]
                dq = np.abs(q - ref)
                de = np.abs(e - ref)
                dqe = np.abs(q - e)
                rms_e = float(np.sqrt((de ** 2).mean()))
                good = rms_e <= a.rms and de.max() <= a.max
                print(f"[{k}] {path.name} scale {scale:g} ({ow}x{oh}): exact vs cv2: max {de.max():.2e}, rms {rms_e:.2e}, "
                      f"identical {np.mean(de == 0) * 100:.1f}%  |  OpenCV-4-quantized vs cv2: max {dq.max():.2e}, rms {np.sqrt((dq ** 2).mean()):.2e}  |  "
                      f"{res['exact'][1]}  {'OK' if good else 'MISMATCH'}", flush=True)
                ok = ok and good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
