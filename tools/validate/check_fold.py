"""Step 3 check: stackcli's pass-1 fold against fusion.fold / fusion.merge.

    python tools/validate/check_fold.py build/cli/stackcli.exe Q:\\TIFF [--report tests/data/report_2026-11-05.json]
                                        [--frames 0,54,81,108,162] [--scale 0.5] [--full-stack]

1. Reference: folds the chosen frames on the CPU (load_aligned + fusion.fold,
   a few seconds per frame) and compares every field of the state with the
   GPU's merged state over the same frames, run on all GPUs.
2. Invariant (--full-stack, GPU only): the whole stack folded on GPU 0, on
   GPU 1, and on both with 3 streams must give byte-identical B and C state
   and A sums equal to double rounding (test_fusion's multi-GPU invariant).
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


def write_list(path, rep, frames_dir):
    with open(path, "w", encoding="utf-8") as f:
        for p in rep["per_frame"]:
            z = p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"]))
            f.write(f"{Path(frames_dir) / p['file']} {z.real:.17g} {z.imag:.17g} {p['tx']:.17g} {p['ty']:.17g}\n")


def run_fold(cli, list_path, rect, scale, out, frames=None, gpus=None, streams=2, extra=()):
    args = [cli, "fold", str(list_path), "--rect", ",".join(map(str, rect)), "--scale", f"{scale:g}", "--out", str(out),
            "--streams", str(streams)] + list(extra)
    if frames is not None:
        args += ["--frames", ",".join(map(str, frames))]
    if gpus is not None:
        args += ["--gpus", ",".join(map(str, gpus))]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"stackcli fold failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def load_state(d):
    d = Path(d)
    man = {}
    for line in (d / "manifest.txt").read_text().split("\n"):
        if line.strip():
            k, v = line.split(" ", 1)
            man[k] = v
    w, h, levels = int(man["w"]), int(man["h"]), int(man["levels"])
    shapes = [tuple(int(x) for x in s.split("x")) for s in man["shapes"].split()]
    st = {"w": w, "h": h, "levels": levels, "count": int(man["count"]), "noise_k": int(man["noise_k"]),
          "noise_fm": float(man["noise_fm"]), "noise_mad": [float(x) for x in man["noise_mad"].split()]}
    st["a_num"] = np.fromfile(d / "a_num.f64", np.float64).reshape(h, w, 3)
    st["a_den"] = np.fromfile(d / "a_den.f64", np.float64).reshape(h, w)
    st["b_best"] = np.fromfile(d / "b_best.f32", np.float32).reshape(h, w)
    st["b_idx"] = np.fromfile(d / "b_idx.i16", np.int16).reshape(h, w)
    st["b_col"] = np.fromfile(d / "b_col.f32", np.float32).reshape(h, w, 3)
    st["b_fmsum"] = np.fromfile(d / "b_fmsum.f64", np.float64).reshape(h, w)
    st["c_sal"], st["c_coef"], st["c_idx"] = [], [], []
    for l in range(levels - 1):
        lw, lh = shapes[l]
        st["c_sal"].append(np.fromfile(d / f"c_sal_{l}.f32", np.float32).reshape(lh, lw))
        st["c_coef"].append(np.fromfile(d / f"c_coef_{l}.f32", np.float32).reshape(lh, lw, 3))
        st["c_idx"].append(np.fromfile(d / f"c_idx_{l}.i16", np.int16).reshape(lh, lw))
    tw, th = shapes[-1]
    st["c_base"] = np.fromfile(d / "c_base.f64", np.float64).reshape(th, tw, 3)
    return st


def rel(a, b, floor=1e-12):
    """max relative difference; `floor` keeps rounding noise on near-zero values from counting"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)))


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def interior(a, m=32):
    return a[m:-m, m:-m]


def compare_states(gpu, ref, levels, tol_rel=1e-3):
    """Tolerances: the focus measure (two Gaussian blurs + Laplacian in float) and the
    pyramid are OpenCV float paths reproduced to summation order, not bit for bit:
    rel 1e-3 on fm-derived sums, 1e-4 abs on colours (the warp's own 7e-5 max)."""
    ok = True

    def line(name, good, text):
        nonlocal ok
        ok = ok and good
        print(f"  {name:12s} {text}  {'OK' if good else 'MISMATCH'}", flush=True)

    r = rel(gpu["a_den"], ref.a_den); ri = rel(interior(gpu["a_den"]), interior(ref.a_den))
    line("a_den", r <= tol_rel, f"max rel {r:.2e} (interior {ri:.2e})")
    A_g = gpu["a_num"] / np.maximum(gpu["a_den"], 1e-300)[:, :, None]
    A_r = ref.a_num / np.maximum(ref.a_den, 1e-300)[:, :, None]
    d = float(np.max(np.abs(A_g - A_r))); line("A", d <= 1e-4, f"weighted average max |diff| {d:.2e}, rms {rms(A_g, A_r):.2e}")
    r = rel(gpu["b_fmsum"], ref.b_fmsum); line("b_fmsum", r <= tol_rel, f"max rel {r:.2e}")
    r = rel(gpu["b_best"], ref.b_best); ri = rel(interior(gpu["b_best"]), interior(ref.b_best))
    line("b_best", r <= tol_rel, f"max rel {r:.2e} (interior {ri:.2e}), rms {rms(gpu['b_best'], ref.b_best):.2e}")
    agree = gpu["b_idx"] == ref.b_idx
    pct = 100 * agree.mean()
    # where the choice differs, the two candidates must have been a near-tie in the reference's own terms
    diff_ok = True
    if not agree.all():
        rb = ref.b_best[~agree]; gb = gpu["b_best"][~agree]
        diff_ok = rel(rb, gb) <= 1e-3
    line("b_idx", pct >= 99.5 and diff_ok, f"agree {pct:.3f}% ({(~agree).sum()} px differ; near-ties: {diff_ok})")
    m = agree[:, :, None].repeat(3, 2)
    d = float(np.max(np.abs(gpu["b_col"][m] - ref.b_col[m]))) if m.any() else 0.0
    line("b_col", d <= 1e-4, f"max |diff| where idx agrees {d:.2e}")
    for l in range(levels - 1):
        # salience = mean of luma^2 over 3x3: a coefficient error e moves it by ~2 sqrt(sal) e, so
        # the warp's 1e-4 colour tolerance allows |diff| <= 2 sqrt(sal) 1e-4 (+ rounding), not a relative bound
        g_, r_ = gpu["c_sal"][l].astype(np.float64), ref.c_sal[l].astype(np.float64)
        allowed = 2 * np.sqrt(np.maximum(g_, r_)) * 1e-4 + 1e-9 + 1e-3 * np.maximum(g_, r_)
        bad = np.abs(g_ - r_) > allowed
        where = ""
        if bad.any():   # where are the bad pixels? (edges tell which border rule is off)
            ys, xs = np.nonzero(bad)
            h_, w_ = bad.shape
            where = (f" | bad {bad.sum()}: left {np.sum(xs == 0)}, right {np.sum(xs == w_ - 1)}, top {np.sum(ys == 0)}, "
                     f"bottom {np.sum(ys == h_ - 1)}, x range {xs.min()}-{xs.max()}, y range {ys.min()}-{ys.max()}")
        line(f"c_sal[{l}]", not bad.any(), f"max |diff| {np.max(np.abs(g_ - r_)):.2e}, rms {rms(g_, r_):.2e}{where}")
        agree = gpu["c_idx"][l] == ref.c_idx[l]
        pct = 100 * agree.mean()
        line(f"c_idx[{l}]", pct >= 99.5, f"agree {pct:.3f}% ({(~agree).sum()} px differ)")
        m = agree[:, :, None].repeat(3, 2)
        d = float(np.max(np.abs(gpu["c_coef"][l][m] - ref.c_coef[l][m]))) if m.any() else 0.0
        line(f"c_coef[{l}]", d <= 1e-4, f"max |diff| where idx agrees {d:.2e}")
    r = rel(gpu["c_base"], ref.c_base); line("c_base", r <= 1e-5, f"max rel {r:.2e}")
    line("count", gpu["count"] == ref.count, f"{gpu['count']} vs {ref.count}")
    line("noise_k", gpu["noise_k"] == ref.noise_k, f"{gpu['noise_k']} vs {ref.noise_k}")
    r = rel(gpu["noise_mad"], ref.noise_mad); line("noise_mad", r <= 1e-3, f"max rel {r:.2e}  gpu {np.array(gpu['noise_mad'])[:3]} ref {np.array(ref.noise_mad)[:3]}")
    r = rel([gpu["noise_fm"]], [ref.noise_fm]); line("noise_fm", r <= 1e-3, f"rel {r:.2e}  gpu {gpu['noise_fm']:.6g} ref {ref.noise_fm:.6g}")
    return ok


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stackcli")
    ap.add_argument("frames_dir")
    ap.add_argument("--report", default=str(REPO / "tests" / "data" / "report_2026-11-05.json"))
    ap.add_argument("--frames", default="0,54,81,108,162")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--full-stack", action="store_true", help="also run the 1-vs-2 GPU invariant on every frame (GPU only)")
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
    ok = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lst = td / "list.txt"
        write_list(lst, rep, a.frames_dir)

        print(f"== reference check: frames {frames} at scale {a.scale:g} ({ow}x{oh}, {levels} levels, sigma {radius})")
        out = run_fold(a.stackcli, lst, rect, a.scale, td / "gpu", frames=frames)
        print("  " + [l for l in out.splitlines() if "frames in" in l][0].strip())
        gpu = load_state(td / "gpu")
        ref = fusion.new_state(oh, ow, levels)
        for k in frames:
            p = pf[k]
            z = p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"]))
            img = fusion.load_aligned(Path(a.frames_dir) / p["file"], z, complex(p["tx"], p["ty"]), grid)
            fusion.fold(ref, k, img, levels, radius, 4.0)
        ok = compare_states(gpu, ref, levels) and ok

        if a.full_stack:
            print("== invariant: whole stack on GPU 0 / GPU 1 / both GPUs x 3 streams")
            runs = {}
            for name, gpus, streams in (("gpu0", [0], 2), ("gpu1", [1], 2), ("both", [0, 1], 3)):
                try:
                    out = run_fold(a.stackcli, lst, rect, a.scale, td / name, gpus=gpus, streams=streams)
                except RuntimeError as e:
                    print(f"  {name}: {e}")
                    continue
                print("  " + name + ": " + [l for l in out.splitlines() if "frames in" in l][0].strip())
                runs[name] = load_state(td / name)
            names = list(runs)
            base = runs[names[0]]
            for n in names[1:]:
                s = runs[n]
                exact = all(same_bytes(base[k], s[k]) for k in ("b_best", "b_idx", "b_col", "b_fmsum"))
                exact_c = all(same_bytes(base["c_sal"][l], s["c_sal"][l]) and same_bytes(base["c_idx"][l], s["c_idx"][l])
                              and same_bytes(base["c_coef"][l], s["c_coef"][l]) for l in range(levels - 1))
                ra = max(rel(base["a_num"], s["a_num"]), rel(base["a_den"], s["a_den"]), rel(base["c_base"], s["c_base"]))
                noise = base["noise_mad"] == s["noise_mad"] and base["noise_fm"] == s["noise_fm"] and base["noise_k"] == s["noise_k"]
                good = exact and exact_c and ra <= 1e-12 and noise
                ok = ok and good
                print(f"  {names[0]} vs {n}: B state identical {exact}, C state identical {exact_c}, A/base max rel {ra:.1e}, "
                      f"noise identical {noise}  {'OK' if good else 'MISMATCH'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
