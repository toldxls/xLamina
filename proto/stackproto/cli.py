"""stackproto command line.

  python -m stackproto synth OUT [--preset NAME] [--set key=value ...]
  python -m stackproto analyze FOLDER --step-um X [--fov-mm Y] [options]
  python -m stackproto validate SYNTH_FOLDER [analyze options]
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import fields
from pathlib import Path

from . import pipeline
from .synth import SynthParams, generate

ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "ideal": {},
    # near-telecentric objective, slightly imperfect train (D ~ 2 m), small tilt
    "telecentric": dict(eps_per_mm=0.0005, tilt_deg=(0.05, -0.02), jitter_um=0.05, noise=0.003),
    # the "everything wrong" case: D = 100 mm, decentred axis (-3.75% of half
    # width), rail tilt, roll, jitter, lead-screw error, flicker
    "faulty": dict(eps_per_mm=0.01, axis_px=(-32.6, 0.0), tilt_deg=(0.25, -0.12),
                   roll_deg_per_mm=0.05, jitter_um=0.15, periodic_um=0.3,
                   periodic_period_um=60.0, flicker=0.01, noise=0.004),
    # reversed / macro lens: D ~ 50 mm
    "macro": dict(eps_per_mm=0.02, tilt_deg=(0.1, 0.05), jitter_um=0.1, noise=0.003),
}


def _parse_set(items):
    names = {f.name for f in fields(SynthParams)}
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown synth param {k!r}; known: {sorted(names)}")
        out[k] = ast.literal_eval(v)
    return out


def _tags(items):
    return dict(it.split("=", 1) for it in items or [])


def _add_analyze_opts(ap):
    ap.add_argument("--step-um", type=float)
    ap.add_argument("--fov-mm", type=float, help="horizontal FOV of one frame (AutoFOV)")
    ap.add_argument("--um-per-px", type=float)
    ap.add_argument("--align-scale", type=float, default=0.25)
    ap.add_argument("--baselines", default="1,2,4,8")
    ap.add_argument("--ref", default="middle",
                    help="first|middle|last|most-zoomed|least-zoomed|<index>")
    ap.add_argument("--policy", default="native-crop", choices=["native-crop", "native-full", "rescaled"])
    ap.add_argument("--axis-px", default="0,0", help="optical axis offset from centre, px (x,y)")
    ap.add_argument("--axis-pct", default=None,
                    help="optical axis as %% of half-width/half-height, e.g. -3.75,0 (lens centering)")
    ap.add_argument("--direction", default="unknown", choices=["unknown", "toward", "away"],
                    help="does increasing frame index move the camera toward the subject?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--tag", action="append", help="history tag key=value (objective, lock, ...)")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--glob", default="*", help="frame filename pattern, e.g. 'IMG_*.tif'")
    ap.add_argument("--lock-scale", action="store_true",
                    help="apply s = 1 in the output geometry (telecentric optics)")
    ap.add_argument("--target", default="specimen", choices=["specimen", "flat"],
                    help="flat = flat test target: diagnostics are trustworthy")


def _analyze(a, folder, step_um=None, fov_mm=None):
    import cv2
    from .frames import list_frames
    step = a.step_um or step_um
    if step is None:
        raise SystemExit("--step-um is required")
    axis = tuple(float(v) for v in a.axis_px.split(","))
    if a.axis_pct:
        first = cv2.imread(str(list_frames(folder, a.glob)[0]), cv2.IMREAD_UNCHANGED)
        H, W = first.shape[:2]
        px, py = (float(v) for v in a.axis_pct.split(","))
        axis = (px / 100 * W / 2, py / 100 * H / 2)
    return pipeline.analyze_folder(
        folder, step, fov_mm=a.fov_mm or fov_mm, um_per_px=a.um_per_px,
        align_scale=a.align_scale, baselines=tuple(int(b) for b in a.baselines.split(",")),
        ref=a.ref, policy=a.policy, axis_px=axis, direction=a.direction, out_dir=a.out,
        png_path=a.png, history_csv=None if a.no_history else ROOT / "history" / "diagnostics.csv",
        tags=_tags(a.tag), workers=a.workers, lock_scale=a.lock_scale, target=a.target,
        pattern=a.glob)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stackproto")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic stack with known faults")
    s.add_argument("out")
    s.add_argument("--preset", default="ideal", choices=sorted(PRESETS))
    s.add_argument("--set", action="append", help="override a SynthParams field, key=value")

    an = sub.add_parser("analyze", help="alignment + geometry report + optics diagnostics")
    an.add_argument("folder")
    _add_analyze_opts(an)

    v = sub.add_parser("validate", help="analyze a synthetic stack and score it against truth")
    v.add_argument("folder")
    _add_analyze_opts(v)

    c = sub.add_parser("compare", help="measure an external stacker's output (e.g. Helicon) "
                                       "against a previous 'analyze' run")
    c.add_argument("analysis_dir", help="the <stack>_stacker folder from analyze")
    c.add_argument("outputs", nargs="+", help="external stacked TIFF/JPEG files")

    fu = sub.add_parser("fuse", help="fuse a stack with methods A/B/C + variants using an "
                                     "'analyze' run's transforms")
    fu.add_argument("analysis_dir")
    fu.add_argument("frames_dir")
    fu.add_argument("--scale", type=float, default=0.5)
    fu.add_argument("--lock-scale", action="store_true")
    fu.add_argument("--policy", default="native-crop", choices=["native-crop", "native-full"])
    fu.add_argument("--roi", default=None, help="x0,y0,x1,y1 in reference full-res px")
    fu.add_argument("--workers", type=int, default=6)
    fu.add_argument("--smoothing", type=int, default=8, help="depth smoothing radius, full-res px")
    fu.add_argument("--radius", type=float, default=6.0, help="focus window, full-res px")
    fu.add_argument("--save-pyramids", action="store_true",
                    help="keep the B and C pyramids so 'blend' can re-render the B<->C slider instantly")

    bl = sub.add_parser("blend", help="B<->C slider from a fuse run's saved pyramids "
                                      "(0 = pure B, 0.5 = B tones + C fine detail, 1 = noise-aware C)")
    bl.add_argument("fusion_dir")
    bl.add_argument("t", type=float, nargs="+", help="slider position(s) in [0, 1]")

    a = ap.parse_args(argv)
    if a.cmd == "synth":
        params = SynthParams(**{**PRESETS[a.preset], **_parse_set(a.set)})
        print(f"synth {a.preset}: {params}")
        generate(a.out, params)
    elif a.cmd == "analyze":
        _analyze(a, a.folder)
    elif a.cmd == "compare":
        import glob
        import numpy as np
        from . import compare
        rep = json.loads((Path(a.analysis_dir) / "report.json").read_text())
        npy = sorted(glob.glob(str(Path(a.analysis_dir) / "cache" / "luma_*.npy")))[-1]
        st = np.load(npy, mmap_mode="r")
        W, H = rep["geometry"]["input"]["width"], rep["geometry"]["input"]["height"]
        res = []
        for f in a.outputs:
            out, _, _ = compare.register_external(f, st, rep, (st.shape[2] / W, st.shape[1] / H))
            res.append(out)
            print(json.dumps(out, indent=1))
        (Path(a.analysis_dir) / "compare.json").write_text(json.dumps(res, indent=1))
    elif a.cmd == "fuse":
        import time
        import numpy as np
        from . import fusion
        from .frames import write_rgb16
        rep = json.loads((Path(a.analysis_dir) / "report.json").read_text())
        pf = rep["per_frame"]
        W, H = rep["geometry"]["input"]["width"], rep["geometry"]["input"]["height"]
        z = np.array([p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"])) for p in pf])
        if a.lock_scale:
            z = z / np.abs(z)
        t = np.array([complex(p["tx"], p["ty"]) for p in pf])
        if a.roi:
            rect = tuple(int(v) for v in a.roi.split(","))
        elif a.policy == "native-crop":
            cp = rep["geometry"]["policies"]["native-crop"]["crop_px"]
            rect = (cp["left"], cp["top"], W - 1 - cp["right"], H - 1 - cp["bottom"])
        else:
            rect = (0, 0, W - 1, H - 1)
        grid = fusion.Grid(W, H, rect, a.scale)
        paths = [Path(a.frames_dir) / p["file"] for p in pf]
        tag = f"fusion_{a.scale:g}" + ("_locked" if a.lock_scale else "") + ("_roi" if a.roi else "")
        od = Path(a.analysis_dir) / tag
        od.mkdir(exist_ok=True)
        t0 = time.time()
        out, info = fusion.run(paths, z, t, grid, workers=a.workers, smoothing_px=a.smoothing,
                               radius_px=a.radius)
        for k, img in out.items():
            write_rgb16(od / f"{k}.tif", img)
        np.save(od / "depth.npy", info["depth"])
        if a.save_pyramids:
            import pickle
            pickle.dump(info["pyramids"], open(od / "pyramids.pkl", "wb"), protocol=4)
        print(f"fused {len(paths)} frames onto {grid.size[0]}x{grid.size[1]} in "
              f"{time.time() - t0:.0f}s -> {od}")
    elif a.cmd == "blend":
        import pickle
        from . import fusion
        from .frames import write_rgb16
        pyr = pickle.load(open(Path(a.fusion_dir) / "pyramids.pkl", "rb"))
        for tv in a.t:
            tv = min(1.0, max(0.0, tv))
            path = Path(a.fusion_dir) / f"slider_{tv:.2f}.tif"
            write_rgb16(path, fusion.blend_bc(pyr["B"], pyr["Cn"], tv))
            print(path)
    elif a.cmd == "validate":
        truth = json.loads((Path(a.folder) / "truth.json").read_text())
        p = truth["params"]
        a.no_history = True
        report, A = _analyze(a, a.folder, step_um=p["step_um"], fov_mm=p["fov_mm"])
        print(json.dumps(pipeline.validate_synth(a.folder, report, A), indent=1))


if __name__ == "__main__":
    main()
