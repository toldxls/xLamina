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
    elif a.cmd == "validate":
        truth = json.loads((Path(a.folder) / "truth.json").read_text())
        p = truth["params"]
        a.no_history = True
        report, A = _analyze(a, a.folder, step_um=p["step_um"], fov_mm=p["fov_mm"])
        print(json.dumps(pipeline.validate_synth(a.folder, report, A), indent=1))


if __name__ == "__main__":
    main()
