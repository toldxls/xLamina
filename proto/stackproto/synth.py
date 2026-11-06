"""Synthetic focus stacks with KNOWN optical-train faults -- ground truth for A6/A8.

Physical model (object space, frame 0 = reference grid, z = rail travel):

* specimen = sharp texture + height map h(x, y) in um spanning the travel;
* frame k focuses at z_k = k * step; a point at height h is blurred with a
  Gaussian of sigma = NA * |h - z_k| / (um per px)  (layered approximation);
* geometry: image of reference point p in frame k is
      p_k = s_k R_k (p - a) + a + d_k
  with s_k = exp(eps * z_k[mm]) zooming about the optical axis point a,
  R_k = roll, and d_k = rail-tilt drift + random jitter + lead-screw periodic
  error (all converted from object um to px).
* photometric: exposure flicker (gain) and signal-dependent sensor noise.

Writes 16-bit RGB TIFF frames plus ``truth.json``, ``_truth_sharp.tif``
(all-in-focus on the frame-0 grid) and ``_truth_depth.npy`` (um).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from . import sim
from .frames import write_rgb16


@dataclass
class SynthParams:
    width: int = 1740
    height: int = 1160
    n_frames: int = 120
    step_um: float = 2.0
    fov_mm: float = 2.0              # horizontal FOV -> um per px
    na: float = 0.28
    eps_per_mm: float = 0.0          # d ln(scale) / d travel, per mm
    axis_px: tuple = (0.0, 0.0)      # optical-axis (zoom-centre) offset from frame centre, px
    tilt_deg: tuple = (0.0, 0.0)     # rail vs optical axis (x, y) -> lateral object drift
    roll_deg_per_mm: float = 0.0
    jitter_um: float = 0.0           # RMS random lateral error per frame, object um
    periodic_um: float = 0.0         # lead-screw periodic lateral error amplitude (x), object um
    periodic_period_um: float = 500.0
    flicker: float = 0.0             # RMS exposure gain error
    noise: float = 0.003             # sensor noise sigma at mid grey (fraction of full scale)
    relief: str = "dome"             # dome | flat | tilt (specimen height map shape)
    layers: int = 28
    seed: int = 1

    @property
    def um_per_px(self) -> float:
        return self.fov_mm * 1000.0 / self.width


def truth_transforms(p: SynthParams, rng: np.random.Generator):
    """Per-frame (z, t) arrays mapping frame-0 coords -> frame-k coords (centred px)."""
    n = p.n_frames
    zk_um = np.arange(n) * p.step_um
    zmm = zk_um / 1000.0
    s = np.exp(p.eps_per_mm * zmm)
    th = np.deg2rad(p.roll_deg_per_mm * zmm)
    zc = s * np.exp(1j * th)
    a = complex(*p.axis_px)
    umpp = p.um_per_px
    drift = (math.tan(math.radians(p.tilt_deg[0])) + 1j * math.tan(math.radians(p.tilt_deg[1]))) \
        * zk_um / umpp
    jit = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (p.jitter_um / umpp / math.sqrt(2))
    jit[0] = 0
    per = p.periodic_um / umpp * (np.sin(2 * math.pi * zk_um / p.periodic_period_um))
    d = drift + jit + per
    t = a - zc * a + d
    return zk_um, zc, t


def _noise_field(rng, H, W, period_px, ch):
    sh, sw = max(2, int(H / period_px) + 3), max(2, int(W / period_px) + 3)
    small = rng.standard_normal((sh, sw, ch)).astype(np.float32)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    return big if ch > 1 else big[:, :, None]


def make_texture(rng, H, W) -> np.ndarray:
    """Sharp RGB specimen texture in ~[0.05, 0.95] with some low-detail zones."""
    coarse = sum(_noise_field(rng, H, W, s, 3) * (s / 64.0) ** 0.5 for s in (32, 64, 128, 256))
    fine = sum(_noise_field(rng, H, W, s, 1) * (s / 8.0) ** 0.35 for s in (1.5, 3, 6, 12))
    zone = 1.0 / (1.0 + np.exp(-3.0 * _noise_field(rng, H, W, 300, 1)))   # 0..1 detail mask
    lum = 0.6 * coarse.mean(axis=2, keepdims=True) + 0.9 * fine * (0.15 + 0.85 * zone)
    tint = 0.25 * coarse
    img = lum + tint
    # dots and thin "hairs"
    canvas = np.zeros((H, W), np.float32)
    for _ in range(int(H * W / 6000)):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(canvas, (x, y), int(rng.integers(1, 4)), float(rng.choice([-1.0, 1.0])),
                   -1, cv2.LINE_AA)
    for _ in range(int(H * W / 40000)):
        x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
        ang = rng.uniform(0, math.pi)
        ln = rng.uniform(40, 300)
        x1, y1 = int(x0 + ln * math.cos(ang)), int(y0 + ln * math.sin(ang))
        cv2.line(canvas, (x0, y0), (x1, y1), -1.5, 1, cv2.LINE_AA)
    img = img + 0.6 * canvas[:, :, None]
    img = (img - img.mean()) / (img.std() + 1e-6)
    return (0.5 + 0.45 * np.tanh(0.55 * img)).astype(np.float32)


def make_height(rng, H, W, zmin, zmax, relief="dome") -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if relief == "flat":                       # flat target at mid-travel
        return np.full((H, W), 0.5 * (zmin + zmax), np.float32)
    if relief == "tilt":                       # tilted flat target
        return (zmin + (zmax - zmin) * xx / (W - 1)).astype(np.float32)
    h = 0.15 * (xx / W) + 0.1 * (yy / H)
    for _ in range(5):
        cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        sg = rng.uniform(0.12, 0.4) * W
        h += rng.uniform(0.4, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg))
    h = (h - h.min()) / (h.max() - h.min())
    return (zmin + (zmax - zmin) * h).astype(np.float32)


def generate(out_dir, p: SynthParams, verbose=True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(p.seed)
    W, H = p.width, p.height
    zk_um, zc, t = truth_transforms(p, rng)

    # canvas big enough for every frame's footprint in frame-0 coords
    c = sim.centre(W, H)
    corners = np.array([0, W - 1, (W - 1) + 1j * (H - 1), 1j * (H - 1)]) - c
    iz, it = sim.invert(zc, t)
    fp = iz[:, None] * corners[None, :] + it[:, None]
    excess = max(np.max(np.abs(fp.real)) - c.real, np.max(np.abs(fp.imag)) - c.imag, 0.0)
    # margin must also exceed the largest defocus blur, or mirror padding at the
    # canvas edge leaks fake (mirrored) content into the frame edges -- that was
    # a measured +-30 ppm/pair edge bias in early runs
    sig_max = p.na * (p.n_frames - 1) * p.step_um / p.um_per_px
    m = int(math.ceil(excess)) + 16 + int(math.ceil(3 * sig_max))
    CH, CW = H + 2 * m, W + 2 * m

    tex = make_texture(rng, CH, CW)
    Z = zk_um[-1]
    height = make_height(rng, CH, CW, 0.06 * Z, 0.94 * Z, p.relief)
    L = p.layers
    lv = np.linspace(height.min(), height.max() + 1e-3, L)
    pos = (height - lv[0]) / (lv[-1] - lv[0]) * (L - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, L - 2)
    fr = (pos - i0).astype(np.float32)
    layers = []
    for l in range(L):
        w = np.where(i0 == l, 1 - fr, 0) + np.where(i0 == l - 1, fr, 0)
        w = w.astype(np.float32)
        layers.append(np.concatenate([tex * w[:, :, None], w[:, :, None]], axis=2))

    umpp = p.um_per_px
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
    gains = 1.0 + p.flicker * rng.standard_normal(p.n_frames)
    # Exact Gaussian defocus in the frequency domain. Blur is linear, so every
    # layer is transformed once and each frame is ONE inverse FFT of
    # sum_l F_l * G(sigma_lk). (A downsample/upsample shortcut used earlier
    # stretched the image by a sigma-dependent amount: fake scale/roll.)
    # Periodic wrap stays inside the canvas margin (>= 3 sigma_max).
    FY = sfft.fftfreq(CH).astype(np.float32)[:, None]
    FX = sfft.rfftfreq(CW).astype(np.float32)[None, :]
    R2 = (2.0 * np.pi ** 2) * (FX * FX + FY * FY)
    spec = [sfft.rfft2(lay, axes=(0, 1), workers=-1) for lay in layers]
    del layers
    for k in range(p.n_frames):
        acc = np.zeros_like(spec[0])
        for l in range(L):
            sg = p.na * abs(lv[l] - zk_um[k]) / umpp
            acc += spec[l] * np.exp(-R2 * np.float32(sg * sg))[:, :, None]
        sc4 = sfft.irfft2(acc, s=(CH, CW), axes=(0, 1), workers=-1)
        scene = sc4[:, :, :3] / np.maximum(sc4[:, :, 3:4], 1e-6)
        # exact float positions (cv2.warpAffine snaps to 1/32 px -- would corrupt truth)
        M = sim.to_affine(iz[k], it[k], W, H)
        X = M[0, 0] * gx + M[0, 1] * gy + M[0, 2] + m
        Y = M[1, 0] * gx + M[1, 1] * gy + M[1, 2] + m
        img = np.stack([ndimage.map_coordinates(
            ndimage.spline_filter(scene[:, :, ch], 3, mode="mirror", output=np.float32),
            [Y, X], order=3, prefilter=False, mode="mirror", output=np.float32)
            for ch in range(3)], axis=2)
        img = img * gains[k]
        sig = p.noise * np.sqrt(0.25 + 1.5 * np.clip(img, 0, 1))
        img = img + sig * rng.standard_normal(img.shape).astype(np.float32)
        write_rgb16(out / f"frame_{k:04d}.tif", img)
        if verbose and (k % 20 == 0 or k == p.n_frames - 1):
            print(f"  synth frame {k + 1}/{p.n_frames}", flush=True)

    write_rgb16(out / "_truth_sharp.tif", tex[m:m + H, m:m + W])
    np.save(out / "_truth_depth.npy", height[m:m + H, m:m + W])
    truth = {
        "params": asdict(p),
        "um_per_px": umpp,
        "z_um": zk_um.tolist(),
        "z": [[v.real, v.imag] for v in zc],
        "t": [[v.real, v.imag] for v in t],
        "gain": gains.tolist(),
        "convention": "T_k maps frame-0 centred px -> frame-k centred px: p_k = z*p + t",
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return out


def load_truth(folder):
    d = json.loads((Path(folder) / "truth.json").read_text())
    z = np.array([complex(*v) for v in d["z"]])
    t = np.array([complex(*v) for v in d["t"]])
    return d, z, t
