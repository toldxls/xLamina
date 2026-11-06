"""End-to-end: folder of frames -> alignment -> geometry report + optical diagnostics."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import align, diagnostics, geometry, sim
from .frames import list_frames, load_luma_stack


def _clean(o):
    """JSON-safe copy: drop keys starting with '_', numpy -> python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def run_alignment(folder, align_scale=0.25, baselines=(1, 2, 4, 8), workers=None,
                  cache_dir=None, log=print):
    paths = list_frames(folder)
    if len(paths) < 3:
        raise SystemExit(f"need >= 3 frames in {folder}, found {len(paths)}")
    t0 = time.time()
    stack, (W, H), (fx, fy) = load_luma_stack(paths, align_scale, cache_dir=cache_dir)
    n = len(paths)
    log(f"loaded {n} frames {W}x{H} -> luma {stack.shape[2]}x{stack.shape[1]} "
        f"in {time.time() - t0:.1f}s")
    npy = getattr(stack, "filename", None)
    if npy is None:
        raise RuntimeError("alignment needs the cached .npy stack (pass cache_dir)")

    t1 = time.time()
    last = [0.0]

    def prog(done, total):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            log(f"  aligned {done}/{total} pairs ({time.time() - t1:.0f}s)")

    pairs = align.align_stack(str(npy), n, baselines, workers, prog)
    z0, tw, info = align.solve_global(n, pairs)
    t_full = tw.real / fx + 1j * tw.imag / fy          # work px -> full-res px
    log(f"global solve: {info['pairs_used']}/{info['pairs_total']} pairs usable, "
        f"{int((~info['inlier']).sum())} flagged as outliers")
    return {"paths": paths, "W": W, "H": H, "z0": z0, "t0": t_full, "pairs": pairs,
            "info": info, "work_factors": (fx, fy)}


def analyze_folder(folder, step_um, fov_mm=None, um_per_px=None, align_scale=0.25,
                   baselines=(1, 2, 4, 8), ref="middle", policy="native-crop",
                   axis_px=(0.0, 0.0), direction="unknown", out_dir=None, png_path=None,
                   history_csv=None, tags=None, workers=None, log=print, lock_scale=False,
                   target="specimen"):
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir else folder.parent / f"{folder.name}_stacker"
    out_dir.mkdir(parents=True, exist_ok=True)
    A = run_alignment(folder, align_scale, baselines, workers, out_dir / "cache", log)
    W, H, z0, t0 = A["W"], A["H"], A["z0"], A["t0"]
    upp = um_per_px if um_per_px else (fov_mm * 1000.0 / W if fov_mm else None)

    diag = diagnostics.analyze(z0, t0, step_um, upp, axis_px, direction)
    diag["target"] = target
    if target != "flat":
        diag["caveat"] = ("specimen stack: measured scale/drift include a relief bias "
                          "(defocused light from other depths, ~NA^2 x surface slope) that can "
                          "exceed the telecentricity signal. Use a FLAT target stack for optics "
                          "diagnostics; consider --lock-scale for stacking with telecentric optics.")
    # geometry uses the APPLIED transforms (scale locked to 1 on request)
    za = z0 / np.abs(z0) if lock_scale else z0
    r = geometry.choose_reference(z0, t0, ref)
    zr, tr = sim.rebase(za, t0, r)
    geo = geometry.report(zr, tr, W, H, upp, r, ref, policy)

    good = A["info"]["good"]
    ncc = np.array([p.ncc for p in good])
    per_frame = [{"i": k, "file": A["paths"][k].name, "s": float(abs(zr[k])),
                  "theta_deg": float(np.degrees(np.angle(zr[k]))),
                  "tx": float(tr[k].real), "ty": float(tr[k].imag)} for k in range(len(zr))]
    report = {
        "stack": folder.name,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "alignment": {"align_scale": align_scale, "baselines": list(baselines),
                      "scale_locked": lock_scale,
                      "pairs_used": A["info"]["pairs_used"], "pairs_total": A["info"]["pairs_total"],
                      "outliers": int((~A["info"]["inlier"]).sum()),
                      "ncc_min": float(ncc.min()), "ncc_median": float(np.median(ncc))},
        "geometry": geo,
        "diagnostics": diag,
        "per_frame": per_frame,
        "transform_convention": "reference-grid centred px -> frame centred px: p_k = s e^{i theta} p + t",
    }
    (out_dir / "report.json").write_text(json.dumps(_clean(report), indent=1))
    title = (f"{folder.name} — {len(zr)} frames, step {step_um} µm, ref {ref} (#{r})"
             + (f", {upp:.4f} µm/px" if upp else ""))
    png = Path(png_path) if png_path else out_dir / "diagnostics.png"
    diagnostics.plot(diag, geo, W, H, zr, tr, png, title)
    if history_csv:
        diagnostics.append_history(history_csv, folder.name, diag, geo, tags or {})
    log(f"report: {out_dir / 'report.json'}\nplot:   {png}")
    return report, A


def validate_synth(folder, report, A):
    """Compare an analysis of a synthetic stack against its truth.json."""
    from .synth import load_truth
    truth, zt, tt = load_truth(folder)
    z0, t0 = A["z0"], A["t0"]
    s_err_ppm = 1e6 * (np.log(np.abs(z0)) - np.log(np.abs(zt)))
    th_err_mdeg = 1e3 * np.degrees(np.angle(z0 / zt))
    # translation error evaluated as point error at the frame corners (worst case)
    W, H = A["W"], A["H"]
    c = sim.centre(W, H)
    corners = np.array([0, W - 1, (W - 1) + 1j * (H - 1), 1j * (H - 1)]) - c
    perr = np.abs((z0[:, None] * corners + t0[:, None]) - (zt[:, None] * corners + tt[:, None]))
    p = truth["params"]
    upp = truth["um_per_px"]
    dtrue = diagnostics.analyze(zt, tt, p["step_um"], upp, tuple(p["axis_px"]))
    dmeas = report["diagnostics"]
    out = {
        "scale_err_ppm_rms": float(np.sqrt(np.mean(s_err_ppm ** 2))),
        "rot_err_mdeg_rms": float(np.sqrt(np.mean(th_err_mdeg ** 2))),
        "corner_err_px_rms": float(np.sqrt(np.mean(perr ** 2))),
        "corner_err_px_max": float(perr.max()),
        "eps_true": dtrue["scale"]["eps_per_mm"], "eps_meas": dmeas["scale"]["eps_per_mm"],
        "eps_se": dmeas["scale"]["eps_se"],
        "eps_injected": p["eps_per_mm"],
        "tilt_true_deg": dtrue.get("tilt", {}).get("angle_deg"),
        "tilt_meas_deg": dmeas.get("tilt", {}).get("angle_deg"),
        "tilt_se_deg": dmeas.get("tilt", {}).get("angle_se_deg"),
        "tilt_injected_deg": list(p["tilt_deg"]),
        "jitter_true_px": dtrue["jitter_px_rms"], "jitter_meas_px": dmeas["jitter_px_rms"],
        "foe_true": dtrue.get("foe_px_from_centre"), "foe_meas": dmeas.get("foe_px_from_centre"),
        "periodic_true": dtrue.get("periodic_x"), "periodic_meas": dmeas.get("periodic_x"),
    }
    # crop from truth vs measured
    geo = report["geometry"]
    r = geo["reference"]["index"]
    ztr, ttr = sim.rebase(zt, tt, r)
    out["crop_true"] = geometry.common_rect(ztr, ttr, W, H)
    out["crop_meas"] = geo["_rect"]
    return _clean(out)
