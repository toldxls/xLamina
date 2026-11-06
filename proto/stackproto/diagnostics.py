"""Optical-train diagnostics from per-frame alignment.

Model (frame 0 = origin, z = rail travel in mm, centred px):
    z_k ~ exp((eps + i*rho) * z)        eps = scale rate (telecentricity residual)
                                        rho = roll rate (rad/mm)
    d_k(q) = T_k(q) - q                 image motion of the optical-axis point q

* eps:   d ln(s)/dz. Implied entrance-pupil distance D = 1/eps (object mm).
         eps = 0 is perfectly object-space telecentric.
* tilt:  image drift of the axis point per mm travel, converted to object um,
         = tan(angle between rail and optical axis). Measured AT q: with the
         wrong q the tilt estimate is off by eps * (q error) per mm.
* FOE:   the point that does not move, -v/(eps + i rho) with v = dt/dz.
         Pinhole optics translating along a tilted rail zoom about the FOE, so
         FOE = axis point + (tilt offset); only measurable when eps >> noise.
         It is NOT the optical axis by itself -- the axis comes from the
         lens-centering (lateral colour) measurement, which is independent.
* jitter: residual of d_k about a linear fit -> rail step repeatability +
         vibration; its spectrum shows lead-screw periodic error.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np


def _linfit(x, y):
    """Least-squares line with slope/intercept standard errors."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef[0], coef[1], math.sqrt(cov[0, 0]), res


def _spectrum(res, step_um):
    n = len(res)
    if n < 16:
        return None
    r = res - res.mean()
    win = np.hanning(n)
    F = np.fft.rfft(r * win)
    amp = 2.0 * np.abs(F) / win.sum()
    freq = np.fft.rfftfreq(n, d=step_um)          # cycles per um
    keep = (freq > 0) & (freq <= 1.0 / (3 * step_um)) & (freq >= 2.0 / (n * step_um))
    if not keep.any():
        return None
    f, a = freq[keep], amp[keep]
    k = int(np.argmax(a))
    return {"period_um": float(1.0 / f[k]), "amplitude": float(a[k]),
            "peak_over_median": float(a[k] / (np.median(a) + 1e-12)),
            "_freq": f, "_amp": a}


def analyze(z, t, step_um, um_per_px=None, axis_px=(0.0, 0.0), direction="unknown"):
    """z, t: per-frame transforms RELATIVE TO FRAME 0, full-res centred px."""
    n = len(z)
    zmm = np.arange(n) * step_um / 1000.0
    lz = np.log(z)
    eps, _, eps_se, res_s = _linfit(zmm, lz.real)
    rho, _, rho_se, res_r = _linfit(zmm, lz.imag)
    q = complex(*axis_px)
    d = z * q + t - q
    vx, _, vx_se, res_x = _linfit(zmm, d.real)
    vy, _, vy_se, res_y = _linfit(zmm, d.imag)
    # translation slope for the FOE (t itself, relative to centre)
    tvx, _, tvx_se, _ = _linfit(zmm, t.real)
    tvy, _, tvy_se, _ = _linfit(zmm, t.imag)

    travel_mm = zmm[-1]
    upp = um_per_px
    out = {
        "frames": n, "step_um": step_um, "travel_mm": travel_mm, "direction": direction,
        "axis_px_from_centre": [q.real, q.imag],
        "scale": {
            "eps_per_mm": eps, "eps_se": eps_se,
            "significant": bool(abs(eps) > 4 * eps_se),
            "total_pct": 100 * (math.exp(eps * travel_mm) - 1),
            "ppm_per_frame": 1e6 * eps * step_um / 1000.0,
            "pupil_distance_mm": (1.0 / eps) if abs(eps) > 4 * eps_se else None,
            "residual_rms_ppm": 1e6 * float(np.sqrt(np.mean(res_s ** 2))),
        },
        "roll": {"deg_per_mm": math.degrees(rho), "se": math.degrees(rho_se),
                 "total_deg": math.degrees(rho * travel_mm)},
        "drift_px_per_mm": [vx, vy], "drift_se_px_per_mm": [vx_se, vy_se],
        "jitter_px_rms": [float(np.std(res_x)), float(np.std(res_y))],
        "jitter_px_max": [float(np.max(np.abs(res_x))), float(np.max(np.abs(res_y)))],
    }
    if direction in ("toward", "away") and out["scale"]["significant"]:
        grows = (eps > 0) == (direction == "toward")
        out["scale"]["kind"] = "entocentric (pupil in front)" if grows else "hypercentric"
    if upp:
        out["tilt"] = {
            "drift_obj_um_per_mm": [vx * upp, vy * upp],
            "angle_deg": [math.degrees(math.atan(vx * upp / 1000.0)),
                          math.degrees(math.atan(vy * upp / 1000.0))],
            "angle_se_deg": [math.degrees(vx_se * upp / 1000.0), math.degrees(vy_se * upp / 1000.0)],
            "axis_sensitivity_um_per_mm_per_px": abs(eps) * upp,
        }
        out["jitter_obj_um_rms"] = [out["jitter_px_rms"][0] * upp, out["jitter_px_rms"][1] * upp]
    lam = complex(eps, rho)
    if out["scale"]["significant"]:
        v = complex(tvx, tvy)
        foe = -v / lam
        rel = math.sqrt((math.hypot(tvx_se, tvy_se) / max(abs(v), 1e-12)) ** 2
                        + (eps_se / abs(eps)) ** 2)
        out["foe_px_from_centre"] = [foe.real, foe.imag]
        out["foe_se_px"] = abs(foe) * rel
    else:
        out["foe_px_from_centre"] = None
        out["foe_note"] = "scale change not significant: system telecentric within noise, FOE undefined"
    spx = _spectrum(res_x, step_um)
    spy = _spectrum(res_y, step_um)
    for key, sp in (("periodic_x", spx), ("periodic_y", spy)):
        if sp:
            out[key] = {k: v for k, v in sp.items() if not k.startswith("_")}
            if upp:
                out[key]["amplitude_obj_um"] = sp["amplitude"] * upp
    out["_series"] = {"zmm": zmm, "ln_s": lz.real, "theta": lz.imag, "dx": d.real, "dy": d.imag,
                      "res_x": res_x, "res_y": res_y, "fit": (eps, rho, vx, vy),
                      "spec_x": spx, "spec_y": spy}
    return out


def plot(diag, geo, W, H, z_ref, t_ref, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .geometry import frame_footprints

    S = diag["_series"]
    zmm = S["zmm"]
    eps, rho, vx, vy = S["fit"]
    upp = geo["input"]["um_per_px_calibrated"]
    unit, k = ("obj µm", upp) if upp else ("px", 1.0)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(title, fontsize=13)

    a = ax[0, 0]
    a.plot(zmm, 1e6 * S["ln_s"], ".", ms=3, label="measured")
    a.plot(zmm, 1e6 * eps * zmm, "-", label=f"fit {1e6 * eps:+.0f} ppm/mm")
    sc = diag["scale"]
    a.set_title(f"Scale vs travel — total {sc['total_pct']:+.4f}% "
                f"({'significant' if sc['significant'] else 'not significant'})")
    a.set_xlabel("rail travel (mm)"); a.set_ylabel("ln s  (ppm)"); a.legend(); a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(zmm, k * S["dx"], ".", ms=3, label="x")
    a.plot(zmm, k * S["dy"], ".", ms=3, label="y")
    a.plot(zmm, k * vx * zmm, "-", c="C0"); a.plot(zmm, k * vy * zmm, "-", c="C1")
    ttl = "Axis-point drift"
    if "tilt" in diag:
        ax_, ay_ = diag["tilt"]["angle_deg"]
        ttl += f" — rail tilt x {ax_:+.3f}°, y {ay_:+.3f}°"
    a.set_title(ttl); a.set_xlabel("rail travel (mm)"); a.set_ylabel(f"drift ({unit})")
    a.legend(); a.grid(alpha=.3)

    a = ax[0, 2]
    a.plot(zmm, 1e3 * np.degrees(S["theta"]), ".", ms=3)
    a.plot(zmm, 1e3 * np.degrees(rho * zmm), "-")
    a.set_title(f"Roll — {diag['roll']['deg_per_mm'] * 1e3:+.2f} mdeg/mm")
    a.set_xlabel("rail travel (mm)"); a.set_ylabel("rotation (mdeg)"); a.grid(alpha=.3)

    a = ax[1, 0]
    a.plot(zmm, k * S["res_x"], "-", lw=.8, label="x")
    a.plot(zmm, k * S["res_y"], "-", lw=.8, label="y")
    jr = diag.get("jitter_obj_um_rms", diag["jitter_px_rms"])
    a.set_title(f"Jitter about the linear fit — RMS x {jr[0]:.3f}, y {jr[1]:.3f} {unit}")
    a.set_xlabel("rail travel (mm)"); a.set_ylabel(unit); a.legend(); a.grid(alpha=.3)

    a = ax[1, 1]
    for key, lab in (("spec_x", "x"), ("spec_y", "y")):
        sp = S[key]
        if sp:
            a.plot(1.0 / sp["_freq"], k * sp["_amp"], "-", lw=.9, label=lab)
    a.set_xscale("log")
    px = diag.get("periodic_x")
    a.set_title("Jitter spectrum (lead-screw periodic error)"
                + (f" — peak {px['period_um']:.0f} µm" if px else ""))
    a.set_xlabel("period (µm of travel)"); a.set_ylabel(f"amplitude ({unit})")
    a.legend(); a.grid(alpha=.3, which="both")

    a = ax[1, 2]
    cov = geo["_coverage"]
    a.imshow(cov, extent=(0, W, H, 0), cmap="viridis", vmin=0, vmax=max(1, cov.max()))
    fp = frame_footprints(z_ref, t_ref, W, H)
    for idx, col in ((0, "w"), (len(fp) - 1, "orange")):
        poly = np.append(fp[idx], fp[idx][0])
        a.plot(poly.real, poly.imag, "-", c=col, lw=1, label=f"frame {idx}")
    x0, y0, x1, y1 = geo["_rect"]
    a.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "r-", lw=1.5, label="native-crop")
    nc = geo["policies"]["native-crop"]
    cp = nc["crop_px"]
    a.set_title(f"Coverage + crop: {W}×{H} → {nc['width']}×{nc['height']}  "
                f"(L{cp['left']} R{cp['right']} T{cp['top']} B{cp['bottom']})")
    a.set_xlim(-0.02 * W, 1.02 * W); a.set_ylim(1.02 * H, -0.02 * H)
    a.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


HISTORY_FIELDS = [
    "timestamp", "stack", "frames", "step_um", "travel_mm", "ref", "um_per_px",
    "eps_per_mm", "eps_se", "scale_total_pct", "corner_shift_px",
    "tilt_x_deg", "tilt_y_deg", "drift_x_px_per_mm", "drift_y_px_per_mm",
    "roll_mdeg_per_mm", "jitter_x_px", "jitter_y_px",
    "periodic_x_period_um", "periodic_x_amp_px", "foe_x_px", "foe_y_px",
    "crop_left", "crop_right", "crop_top", "crop_bottom", "out_w", "out_h", "tags",
]


def append_history(path, stack_name, diag, geo, tags: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nc = geo["policies"]["native-crop"]
    tilt = diag.get("tilt", {}).get("angle_deg", [None, None])
    px = diag.get("periodic_x") or {}
    foe = diag.get("foe_px_from_centre") or [None, None]
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "stack": stack_name,
        "frames": diag["frames"], "step_um": diag["step_um"], "travel_mm": diag["travel_mm"],
        "ref": geo["reference"]["index"], "um_per_px": geo["input"]["um_per_px_calibrated"],
        "eps_per_mm": diag["scale"]["eps_per_mm"], "eps_se": diag["scale"]["eps_se"],
        "scale_total_pct": diag["scale"]["total_pct"],
        "corner_shift_px": geo["scale_spread"]["corner_shift_px"],
        "tilt_x_deg": tilt[0], "tilt_y_deg": tilt[1],
        "drift_x_px_per_mm": diag["drift_px_per_mm"][0], "drift_y_px_per_mm": diag["drift_px_per_mm"][1],
        "roll_mdeg_per_mm": 1e3 * diag["roll"]["deg_per_mm"],
        "jitter_x_px": diag["jitter_px_rms"][0], "jitter_y_px": diag["jitter_px_rms"][1],
        "periodic_x_period_um": px.get("period_um"), "periodic_x_amp_px": px.get("amplitude"),
        "foe_x_px": foe[0], "foe_y_px": foe[1],
        "crop_left": nc["crop_px"]["left"], "crop_right": nc["crop_px"]["right"],
        "crop_top": nc["crop_px"]["top"], "crop_bottom": nc["crop_px"]["bottom"],
        "out_w": nc["width"], "out_h": nc["height"],
        "tags": ";".join(f"{k}={v}" for k, v in sorted(tags.items())),
    }
    new = not path.exists()
    with path.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            wr.writeheader()
        wr.writerow(row)
