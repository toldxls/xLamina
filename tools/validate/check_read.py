"""Step 1 check: stackcli's GPU read path against the Python reference
frames.read_rgb (the CUDA port must reproduce the prototype, CLAUDE.md).

    python tools/validate/check_read.py build/cli/stackcli.exe Q:\\TIFF [--frames 0,81,162] [--sums sums.txt]

Per frame: runs `stackcli dump`, loads the raw float32 RGB and compares it
pixel for pixel with read_rgb (expected: max |diff| <= 1 ulp of float32 at
1.0, i.e. 6e-8). With --sums (from `stackcli stream --sums`), also checks the
listed frames' checksums against numpy's float64 sum of read_rgb.
Only the requested frames are read on the CPU (one 194 MB TIFF each).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "proto"))
from stackproto.frames import list_frames, read_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stackcli")
    ap.add_argument("frames_dir")
    ap.add_argument("--glob", default="IMG_*.tif")
    ap.add_argument("--frames", default="0,81,162", help="indices into the sorted frame list")
    ap.add_argument("--sums", default=None, help="checksum file written by `stackcli stream --sums`")
    ap.add_argument("--tol", type=float, default=1e-7)
    a = ap.parse_args()

    paths = list_frames(a.frames_dir, a.glob)
    idx = [int(v) for v in a.frames.split(",")]
    sums = {}
    if a.sums:
        for line in Path(a.sums).read_text().splitlines():
            name, val = line.rsplit(" ", 1)
            sums[name] = float(val)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for k in idx:
            p = paths[k]
            raw = Path(td) / "frame.f32"
            r = subprocess.run([a.stackcli, "dump", str(p), str(raw)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{k}] {p.name}: stackcli failed\n{r.stdout}{r.stderr}")
                ok = False
                continue
            ref = read_rgb(p)
            got = np.fromfile(raw, np.float32).reshape(ref.shape)
            d = np.abs(got - ref)
            line = f"[{k}] {p.name}: max|diff| {d.max():.3e}, mean {d.mean():.3e}, exact {np.mean(d == 0) * 100:.2f}%"
            good = d.max() <= a.tol
            if p.name in sums:
                s_ref = float(ref.sum(dtype=np.float64))
                rel = abs(sums[p.name] - s_ref) / max(abs(s_ref), 1e-9)
                line += f"; checksum gpu {sums[p.name]:.3f} vs numpy {s_ref:.3f} (rel {rel:.1e})"
                good = good and rel <= 1e-9
            print(line + ("  OK" if good else "  MISMATCH"), flush=True)
            ok = ok and good
            raw.unlink(missing_ok=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
