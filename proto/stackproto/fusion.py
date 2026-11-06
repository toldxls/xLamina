"""Fusion prototypes (A8): methods A, B, C and the halo/noise variants.

Everything is written as ORDER-INDEPENDENT folds -- the shape of the CUDA design:
each worker (GPU) folds its own frames into a partial state, partial states
merge in any order, ties break toward the lower frame index, so the result is
identical however the frames are split (test_fusion checks this).

Methods
-------
A   weighted average: w = (FM + eps)^p, sum(w I) / sum(w).
B   depth map by cost-volume filtering: each frame's focus measure is aggregated
    with a guided filter (guide = A), depth = running argmax (pass 2); a third
    pass composes each pixel from the two frames bracketing its depth. The older
    confidence-threshold estimator is kept as depth_method="heuristic".
C   Laplacian pyramid: per level and pixel keep the coefficient with the highest
    local energy (luma, 3x3); base level = mean.

Variants (post-merge, need B and C)
-----------------------------------
CB  depth-guided pyramid: where C's choice disagrees with B's depth by more than
    `agree` frames and isn't clearly dominant, take B's coefficient instead.
Cn  noise-aware: coefficients below k * sigma_noise(level) come from B (max of
    N noisy samples grows like sigma*sqrt(2 ln N) -- the pyramid noise build-up).
CBn both of the above.
H   hybrid luma/chroma: luma from CBn, chroma from B.
(Tried and rejected: "Cs" = C with global soft-threshold shrinkage + a selected
base. At full res fine needle coefficients are noise-sized, so it erased detail;
synthetic PSNR 34.5 vs C 37.4. See docs/FINDINGS.md.)
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import sim
from .frames import read_rgb

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ----------------------------------------------------------------------------- frames

@dataclass
class Grid:
    """Output grid: a rectangle of the reference frame, at a working scale."""
    W: int
    H: int
    rect: tuple            # (x0, y0, x1, y1) inclusive, reference FULL-res px
    scale: float = 0.5

    @property
    def size(self):
        x0, y0, x1, y1 = self.rect
        return (int(round((x1 - x0 + 1) * self.scale)), int(round((y1 - y0 + 1) * self.scale)))


def load_aligned(path, z, t, grid: Grid):
    """Frame warped onto the output grid. z, t: ref->frame transform (full-res centred)."""
    img = read_rgb(path)
    H, W = img.shape[:2]
    s = grid.scale
    if s != 1.0:
        w, h = int(round(W * s)), int(round(H * s))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        w, h = W, H
    fx, fy = w / W, h / H
    M = sim.to_affine(z, complex(t.real * fx, t.imag * fy), w, h)
    # output pixel (u, v) = reference work pixel (x0*fx + u, y0*fy + v)
    x0, y0 = grid.rect[0] * fx, grid.rect[1] * fy
    M[:, 2] = M[:, 2] + M[:, :2] @ np.array([x0, y0])
    ow, oh = grid.size
    return cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def focus_measure_rgb(img, radius=3.0, normalized=False, floor=0.02):
    """Laplacian energy. ``normalized`` divides by local brightness^2: in a
    featureless region the raw measure rises with brightness (shot noise, glow
    gradients), so the argmax picks the frame where defocused neighbours spill
    the MOST light in -- the B halo. Measured on the real stack: a dark gap's raw
    FM rose 5.9x from frame 0 to 162 while it brightened 3.2x with needle glow."""
    y = img @ LUMA
    lap = cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 0.8), cv2.CV_32F, ksize=3)
    fm = cv2.GaussianBlur(lap * lap, (0, 0), radius)
    if normalized:
        m = cv2.GaussianBlur(y, (0, 0), 3.0 * radius)
        fm = fm / (m * m + floor * floor)
    return fm


# ----------------------------------------------------------------------------- pyramids

def lap_pyramid(img, levels):
    g = [img]
    for _ in range(levels - 1):
        g.append(cv2.pyrDown(g[-1]))
    lap = []
    for l in range(levels - 1):
        up = cv2.pyrUp(g[l + 1], dstsize=(g[l].shape[1], g[l].shape[0]))
        lap.append(g[l] - up)
    lap.append(g[-1])
    return lap


def collapse(lap):
    cur = lap[-1]
    for l in range(len(lap) - 2, -1, -1):
        cur = cv2.pyrUp(cur, dstsize=(lap[l].shape[1], lap[l].shape[0])) + lap[l]
    return cur


def n_levels(w, h, top=32):
    return max(2, int(math.floor(math.log2(min(w, h) / top))) + 1)


def salience(coef):
    y = coef @ LUMA
    return cv2.boxFilter(y * y, -1, (3, 3))


# ----------------------------------------------------------------------------- fold states

@dataclass
class State:
    """Partial result for a subset of frames (one GPU's share)."""
    # A
    a_num: np.ndarray = None
    a_den: np.ndarray = None
    # B pass 1
    b_best: np.ndarray = None
    b_idx: np.ndarray = None
    b_col: np.ndarray = None         # colour of the argmax frame (unsmoothed B)
    b_fmsum: np.ndarray = None
    # C
    c_sal: list = field(default_factory=list)
    c_coef: list = field(default_factory=list)
    c_idx: list = field(default_factory=list)
    c_base: np.ndarray = None
    count: int = 0
    # noise estimate source: pyramid of the frame with the lowest index seen
    noise_k: int = 1 << 30
    noise_mad: list = field(default_factory=list)
    noise_fm: float = 0.0            # median focus measure of that frame = FM noise floor


def new_state(h, w, levels):
    st = State()
    st.a_num = np.zeros((h, w, 3), np.float64)
    st.a_den = np.zeros((h, w), np.float64)
    st.b_best = np.full((h, w), -1.0, np.float32)
    st.b_idx = np.zeros((h, w), np.int16)
    st.b_col = np.zeros((h, w, 3), np.float32)
    st.b_fmsum = np.zeros((h, w), np.float64)
    shapes = [(h, w)]
    for _ in range(levels - 1):
        hh, ww = shapes[-1]
        shapes.append(((hh + 1) // 2, (ww + 1) // 2))
    for l in range(levels - 1):
        hh, ww = shapes[l]
        st.c_sal.append(np.full((hh, ww), -1.0, np.float32))
        st.c_coef.append(np.zeros((hh, ww, 3), np.float32))
        st.c_idx.append(np.zeros((hh, ww), np.int16))
    hh, ww = shapes[-1]
    st.c_base = np.zeros((hh, ww, 3), np.float64)
    return st


def fold(st: State, k: int, img, levels, radius=3.0, power=4.0):
    fm = focus_measure_rgb(img, radius)
    # A
    w = np.power(fm.astype(np.float64) + 1e-9, power / 2)
    st.a_num += w[:, :, None] * img
    st.a_den += w
    # B: strict '>' keeps the lower index on ties when frames arrive in order;
    # merge() applies the same rule across workers
    m = fm > st.b_best
    st.b_best[m] = fm[m]
    st.b_idx[m] = k
    st.b_col[m] = img[m]
    st.b_fmsum += fm
    # C
    lap = lap_pyramid(img, levels)
    for l in range(levels - 1):
        s = salience(lap[l])
        m = s > st.c_sal[l]
        st.c_sal[l][m] = s[m]
        st.c_coef[l][m] = lap[l][m]
        st.c_idx[l][m] = k
    st.c_base += lap[-1]
    st.count += 1
    if k < st.noise_k:          # noise reference: lowest frame index (any split)
        st.noise_k = k
        st.noise_mad = [float(1.4826 * np.median(np.abs(lap[l] @ LUMA))) for l in range(levels - 1)]
        st.noise_fm = float(np.median(fm))


def _take(dst_val, dst_idx, src_val, src_idx):
    """Mask where src wins: higher value, ties -> lower frame index."""
    return (src_val > dst_val) | ((src_val == dst_val) & (src_idx < dst_idx))


def merge(a: State, b: State) -> State:
    a.a_num += b.a_num
    a.a_den += b.a_den
    m = _take(a.b_best, a.b_idx, b.b_best, b.b_idx)
    a.b_best[m] = b.b_best[m]
    a.b_idx[m] = b.b_idx[m]
    a.b_col[m] = b.b_col[m]
    a.b_fmsum += b.b_fmsum
    for l in range(len(a.c_sal)):
        m = _take(a.c_sal[l], a.c_idx[l], b.c_sal[l], b.c_idx[l])
        a.c_sal[l][m] = b.c_sal[l][m]
        a.c_coef[l][m] = b.c_coef[l][m]
        a.c_idx[l][m] = b.c_idx[l][m]
    a.c_base += b.c_base
    a.count += b.count
    if b.noise_k < a.noise_k:
        a.noise_k, a.noise_mad, a.noise_fm = b.noise_k, b.noise_mad, b.noise_fm
    return a


# ----------------------------------------------------------------------------- B composition

def guided_filter(guide, src, r, eps):
    """He et al. guided filter (grey guide), box radius r."""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(guide, -1, k)
    mp = cv2.boxFilter(src, -1, k)
    cIp = cv2.boxFilter(guide * src, -1, k) - mI * mp
    vI = cv2.boxFilter(guide * guide, -1, k) - mI * mI
    a = cIp / (vI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def smooth_depth_map(st: State, smoothing=4, eps=1e-3, fill_sigmas=(4, 16, 64)):
    # all radii are in OUTPUT pixels; run() converts from full-res pixels
    """Depth map for B: confidence -> fill -> despeckle -> edge-aware smoothing.

    Confidence is the peak-to-mean ratio of each pixel's focus curve (see below). The relative-peak confidence used first could not tell
    noise from detail and left blotchy patchworks in defocused backgrounds.
    Low-confidence pixels take their depth from confident neighbours
    (normalised convolution, finest scale with enough support first).
    """
    best = st.b_best.astype(np.float32)
    mean = (st.b_fmsum / max(st.count, 1)).astype(np.float32)
    # peak-to-mean of each pixel's own focus curve: pure noise has a flat curve
    # (ratio ~1.5-2.3 measured in the defocused background), real detail peaks
    # (4-16). A global noise-floor test was tried and rejected: at full res, fine
    # low-contrast detail sits barely above the per-pixel noise floor.
    ratio = best / np.maximum(mean, 1e-12)
    conf = np.clip(np.log(np.maximum(ratio / 1.4, 1.0)) / np.log(6.0 / 1.4), 0.0, 1.0).astype(np.float32)
    d = cv2.medianBlur(st.b_idx.astype(np.float32), 5)
    w = conf * conf
    filled = None
    for sg in fill_sigmas:
        num = cv2.GaussianBlur(d * w, (0, 0), sg)
        den = cv2.GaussianBlur(w, (0, 0), sg)
        est = num / np.maximum(den, 1e-6)
        if filled is None:
            filled = np.where(den > 0.05, est, np.nan)
        else:
            filled = np.where(np.isnan(filled) & (den > 0.01), est, filled)
    glob = float(np.sum(d * w) / max(float(np.sum(w)), 1e-12))
    filled = np.where(np.isnan(filled), glob, filled).astype(np.float32)
    # keep confident pixels' own depth, fill the rest
    dd = conf * d + (1 - conf) * filled
    if smoothing <= 0:
        return dd.astype(np.float32), conf
    # guide with the weighted average (A): smooth, real edges only. Guiding with
    # the argmax colour (B0) preserved its noise-driven patch outlines.
    a_img = (st.a_num / np.maximum(st.a_den, 1e-30)[:, :, None]).astype(np.float32)
    guide = (a_img @ LUMA).astype(np.float32)
    ds = guided_filter(guide, dd.astype(np.float32), smoothing, eps)
    return ds.astype(np.float32), conf


@dataclass
class BCompose:
    num: np.ndarray
    den: np.ndarray


def fold_b2(acc: BCompose, k, img, depth, soft_sigma=None):
    if soft_sigma:
        # soft label compositing: blurred indicator of "label == k" -- the blurred
        # indicators of a partition stay a partition of unity, and a boundary
        # blends the two SIDES' frames
        lab = (np.abs(depth - k) < 0.5).astype(np.float32)
        if not lab.any():
            return
        w = cv2.GaussianBlur(lab, (0, 0), soft_sigma)
    else:
        w = np.clip(1.0 - np.abs(depth - k), 0.0, 1.0)
    if w.any():
        acc.num += w[:, :, None] * img
        acc.den += w


# ----------------------------------------------------------------------------- finalize + variants

def finalize(st: State, depth_s, b_img, agree=3.0, dominance=4.0, noise_k=3.0, coarse_b_level=3,
             glint_agree=6.0):
    out = {}
    out["A"] = (st.a_num / np.maximum(st.a_den, 1e-30)[:, :, None]).astype(np.float32)
    out["B"] = b_img
    out["B0"] = st.b_col.copy()
    base = (st.c_base / max(st.count, 1)).astype(np.float32)
    lapC = st.c_coef + [base]
    out["C"] = collapse(lapC)
    lapB = lap_pyramid(b_img, len(lapC))
    variants = {"CB": (True, False), "Cn": (False, True), "CBn": (True, True)}
    maps = {}
    for name, (use_depth, use_noise) in variants.items():
        lv = []
        replaced = []
        for l in range(len(lapC) - 1):
            cC, cB = lapC[l], lapB[l]
            take_b = np.zeros(cC.shape[:2], bool)
            if use_depth:
                dl = cv2.resize(depth_s, (cC.shape[1], cC.shape[0]), interpolation=cv2.INTER_AREA)
                sB = salience(cB)
                take_b |= (np.abs(st.c_idx[l] - dl) > agree) & (st.c_sal[l] < dominance * sB)
            wb = take_b.astype(np.float32)
            if use_noise:
                # soft: fully B below 0.5*k*sigma, fully C above 1.5*k*sigma
                sig = st.noise_mad[l]
                r = np.abs(cC @ LUMA) / (noise_k * sig + 1e-12)
                wn = np.clip(1.5 - r, 0.0, 1.0)
                wb = np.maximum(wb, wn)
            wb = cv2.GaussianBlur(wb, (0, 0), 1.0)[:, :, None]   # no hard seams
            lv.append(wb * cB + (1 - wb) * cC)
            replaced.append(float(wb.mean()))
        lv.append(lapB[-1])            # base from B: no averaged-in defocus haze
        out[name] = collapse(lv)
        maps[name] = replaced
        if name == "Cn":
            pyr_cn = lv
    # Cd: depth-locked pyramid -- fine-level coefficients only from frames
    # within +-glint_agree of B's depth (specular glints are virtual images of
    # the lamp at OTHER depths; unrestricted max-selection collects them from
    # every frame as sparkle), noise-aware rule otherwise, coarse levels from B.
    lv = []
    for l in range(len(lapC) - 1):
        cC, cB = lapC[l], lapB[l]
        if l >= coarse_b_level:
            lv.append(cB)
            continue
        dl = cv2.resize(depth_s, (cC.shape[1], cC.shape[0]), interpolation=cv2.INTER_AREA)
        far = (np.abs(st.c_idx[l].astype(np.float32) - dl) > glint_agree).astype(np.float32)
        r = np.abs(cC @ LUMA) / (noise_k * st.noise_mad[l] + 1e-12)
        wb = np.maximum(far, np.clip(1.5 - r, 0.0, 1.0))
        wb = cv2.GaussianBlur(wb, (0, 0), 1.0)[:, :, None]
        lv.append(wb * cB + (1 - wb) * cC)
    lv.append(lapB[-1])
    out["Cd"] = collapse(lv)
    # Cg: glare-safe pyramid. Coarse levels (>= coarse_b_level) come from B --
    # max-energy selection there picks large DEFOCUSED bright blobs (big
    # blurry highlights carry the most low-frequency energy), i.e. glow/glare
    # from every frame; fine levels keep the noise-aware selection (Cn rule).
    for L0, name in ((coarse_b_level, "Cg"), (coarse_b_level - 1, "Cg2")):
        lv = []
        for l in range(len(lapC) - 1):
            if l >= L0:
                lv.append(lapB[l])
                continue
            cC, cB = lapC[l], lapB[l]
            r = np.abs(cC @ LUMA) / (noise_k * st.noise_mad[l] + 1e-12)
            wb = cv2.GaussianBlur(np.clip(1.5 - r, 0.0, 1.0), (0, 0), 1.0)[:, :, None]
            lv.append(wb * cB + (1 - wb) * cC)
        lv.append(lapB[-1])
        out[name] = collapse(lv)
    # hybrid: luma from CBn, chroma from B (BT.709 on the encoded values)
    out["H"] = _swap_chroma(out["CBn"], b_img)
    return out, maps, {"B": lapB, "Cn": pyr_cn}


def blend_bc(pyr_b, pyr_c, t):
    """The B <-> C slider. t = 0: pure B; t = 1: pure (noise-aware) C. Levels
    cross over FINE-FIRST: at t = 0.5 the finest detail is all C while the
    coarsest levels and base are still B (B's glare-free tones + C's detail).
    Pyramids are linear, so this re-renders instantly from the two stored
    pyramids -- no re-fusion (a live GUI slider on the GPU)."""
    n = len(pyr_b)
    lv = []
    for l in range(n):
        frac = l / max(n - 1, 1)                  # 0 finest ... 1 base
        w = float(np.clip(2.0 * t - frac, 0.0, 1.0))
        lv.append((1 - w) * pyr_b[l] + w * pyr_c[l])
    return collapse(lv)


_M = np.array([[0.2126, 0.7152, 0.0722], [-0.1146, -0.3854, 0.5], [0.5, -0.4542, -0.0458]],
              np.float32)
_Mi = np.linalg.inv(_M).astype(np.float32)


def _swap_chroma(y_src, c_src):
    a = y_src @ _M.T
    b = c_src @ _M.T
    a[:, :, 1:] = b[:, :, 1:]
    return a @ _Mi.T


# ----------------------------------------------------------------------------- drivers (split mode)

_CTX = {}


def _w_init(ctx):
    _CTX.update(ctx)


def _w_fold(chunk):
    g, levels = _CTX["grid"], _CTX["levels"]
    ow, oh = g.size
    st = new_state(oh, ow, levels)
    for k in chunk:
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        fold(st, k, img, levels, _CTX["radius"], _CTX["power"])
    return st


def _w_b2(chunk):
    g = _CTX["grid"]
    ow, oh = g.size
    acc = BCompose(np.zeros((oh, ow, 3), np.float64), np.zeros((oh, ow), np.float64))
    depth = _CTX["depth"]
    lo, hi = float(depth.min()), float(depth.max())
    for k in chunk:
        if k < lo - 1 or k > hi + 1:
            continue
        if _CTX.get("soft_sigma") and not (np.abs(depth - k) < 0.5).any():
            continue
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        fold_b2(acc, k, img, depth, _CTX.get("soft_sigma"))
    return acc


def _w_agg(chunk):
    """Cost-volume filtering, one worker's frames: edge-aware aggregate of each
    frame's focus measure, running argmax (ties -> lower index, like merge)."""
    g = _CTX["grid"]
    ow, oh = g.size
    guide, r, eps = _CTX["guide"], _CTX["agg_r"], _CTX["agg_eps"]
    best = np.full((oh, ow), -np.inf, np.float32)
    idx = np.zeros((oh, ow), np.int16)
    for k in chunk:
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        agg = guided_filter(guide, focus_measure_rgb(img, _CTX["fm_radius"], normalized=True), r, eps)
        m = agg > best
        best[m] = agg[m]
        idx[m] = k
    return best, idx


def depth_cvf(ctx, chunks, workers, order=None):
    """Depth by cost-volume filtering (Hosni et al. style, focus instead of stereo
    cost): no confidence thresholds -- evidence spreads within an object (the
    guide's edges stop it), so featureless areas inherit their object's depth
    instead of a noise-driven argmax."""
    with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
        parts = list(ex.map(_w_agg, chunks))
    ids = list(order) if order is not None else list(range(len(parts)))
    best, idx = parts[ids[0]]
    best, idx = best.copy(), idx.copy()
    for i in ids[1:]:
        b2, i2 = parts[i]
        m = _take(best, idx, b2, i2)
        best[m] = b2[m]
        idx[m] = i2[m]
    # integer labels; boundaries are blended in composition (soft labels), NOT by
    # blurring the depth -- a blurred jump passes through intermediate frames
    # that have nothing in focus there (dark seams)
    return cv2.medianBlur(idx.astype(np.float32), 5), best


def _w_fmvol(chunk):
    """Normalised focus measure of each frame at the depth resolution."""
    g = _CTX["grid"]
    dw, dh = _CTX["dsize"]
    out = {}
    for k in chunk:
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        # measure at the FIXED depth resolution (a fixed fraction of the sensor),
        # so every threshold means the same at any output scale. (Measuring at
        # the output scale and resizing made full-res runs fall below the
        # half-res-tuned thresholds: washed-out, glowing patches.)
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        fm = focus_measure_rgb(img, _CTX["fm_radius"], normalized=True)
        lum = cv2.GaussianBlur(img @ LUMA, (0, 0), 2.0 * _CTX["fm_radius"])
        out[k] = (fm.astype(np.float16), lum.astype(np.float16))
    return out


def sgm_from_volume(V, Lv, guide_full, ctx):
    """Cost from the focus (V) and brightness (Lv) volumes -> SGM -> depth on the
    output grid. Confidence = contrast of the pixel's focus curve, ramped from
    ``sgm_c_lo`` (0) to ``sgm_c_hi`` (1). Low-confidence pixels follow the
    darkness prior: measured on the real stack, Helicon's background is the
    darkest frame 70 % of the time vs 11 % for the focus argmax, and background
    contrast is 0.39-0.51 against 0.66-0.69 in texture."""
    from .sgm import argmin_subpixel, sgm
    dh, dw, K = V.shape
    V = np.sqrt(V)
    vmax = V.max(axis=2, keepdims=True)
    vmed = np.median(V, axis=2, keepdims=True)
    contrast = (vmax - vmed) / np.maximum(vmax, 1e-12)
    lo, hi = ctx.get("sgm_c_lo", 0.45), ctx.get("sgm_c_hi", 0.60)
    w = np.clip((contrast - lo) / (hi - lo), 0.0, 1.0)
    C = w * (1.0 - V / np.maximum(vmax, 1e-12))
    del V
    lam = ctx.get("dark_prior", 0.0)
    lam_h = ctx.get("halo_prior", 0.0)
    g = cv2.resize(guide_full, (dw, dh), interpolation=cv2.INTER_AREA)
    if lam > 0 or lam_h > 0:
        lmin = Lv.min(axis=2, keepdims=True)
        lmax = Lv.max(axis=2, keepdims=True)
        dark = (Lv - lmin) / np.maximum(lmax - lmin, 0.01)
        if lam > 0:
            C = C + (1.0 - w) * lam * dark
        if lam_h > 0:
            # spill zone: much darker than something bright nearby. Defocused
            # light from that neighbour can only ADD brightness here, so prefer
            # the darkest frame even where the focus curve looks confident (the
            # B edge halo). Bright objects themselves are the local max: h ~ 0.
            r = max(1, int(round(ctx.get("halo_radius_px", 24.0) * ctx.get("depth_full_fraction", 0.25))))
            nbmax = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            h = np.clip(((nbmax - g) / (nbmax + 0.02) - 0.3) / 0.5, 0.0, 1.0)
            C = C + lam_h * h[:, :, None] * dark
        del dark
    C = C.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = np.sqrt(gx * gx + gy * gy)
    P2 = (ctx["sgm_p2"] / (1.0 + grad / ctx["sgm_edge"])).astype(np.float32)
    P2 = np.maximum(P2, 2 * ctx["sgm_p1"])
    S = sgm(C, ctx["sgm_p1"], P2, ctx["sgm_dirs"])
    d = argmin_subpixel(S)
    del S, C
    oh, ow = guide_full.shape
    up = cv2.resize(d, (ow, oh), interpolation=cv2.INTER_LINEAR)
    up = guided_filter(guide_full, up, max(2, int(round(2 * ow / dw))), 1e-4)
    return (cv2.medianBlur(np.rint(up).astype(np.float32), 3),
            cv2.resize(w[:, :, 0].astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR))


def depth_sgm(ctx, chunks, workers, guide_full, log=print):
    """Regularised depth: focus cost volume at reduced resolution -> SGM ->
    sub-frame argmin -> edge-guided upsampling to the output grid."""
    from .sgm import argmin_subpixel, sgm
    dw, dh = ctx["dsize"]
    cache = ctx.get("vol_cache")
    if cache and os.path.exists(cache):
        z_ = np.load(cache)
        V, Lv = z_["fm"].astype(np.float32), z_["lum"].astype(np.float32)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
            parts = list(ex.map(_w_fmvol, chunks))
        vol = {}
        for p in parts:
            vol.update(p)
        K = len(vol)
        V = np.empty((dh, dw, K), np.float32)
        Lv = np.empty((dh, dw, K), np.float32)
        for k in range(K):
            V[:, :, k], Lv[:, :, k] = vol[k]
        del vol, parts
        if cache:
            np.savez(cache, fm=V.astype(np.float16), lum=Lv.astype(np.float16))
    d_up, conf = sgm_from_volume(V, Lv, guide_full, ctx)
    log(f"  depth: SGM on {dw}x{dh}x{V.shape[2]}, {ctx['sgm_dirs']} directions")
    return d_up, conf


def _w_refine(chunk):
    """Fine-scale argmax restricted to +-R frames around the regularised depth."""
    g = _CTX["grid"]
    ow, oh = g.size
    D, R = _CTX["depth"], _CTX["refine_r"]
    best = np.full((oh, ow), -np.inf, np.float32)
    idx = np.zeros((oh, ow), np.int16)
    for k in chunk:
        win = np.abs(D - k) <= R
        if not win.any():
            continue
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        fm = focus_measure_rgb(img, _CTX["refine_fm_r"], normalized=True)
        fm = np.where(win, fm, -np.inf)
        m = fm > best
        best[m] = fm[m]
        idx[m] = k
    return best, idx


def _w_b2dual(chunk):
    """Soft-label composition with a narrow AND a wide label blur (mixed later
    by confidence: crisp in texture, smooth in featureless regions)."""
    g = _CTX["grid"]
    ow, oh = g.size
    D = _CTX["depth"]
    s1, s2 = _CTX["soft_sigma"], _CTX["soft_sigma_wide"]
    acc = [np.zeros((oh, ow, 3), np.float64), np.zeros((oh, ow), np.float64),
           np.zeros((oh, ow, 3), np.float64), np.zeros((oh, ow), np.float64)]
    for k in chunk:
        lab = (np.abs(D - k) < 0.5).astype(np.float32)
        if not lab.any():
            continue
        img = load_aligned(_CTX["paths"][k], _CTX["z"][k], _CTX["t"][k], g)
        for (num, den), sg in (((acc[0], acc[1]), s1), ((acc[2], acc[3]), s2)):
            w = cv2.GaussianBlur(lab, (0, 0), sg)
            num += w[:, :, None] * img
            den += w
    return acc


def _chunks(n, parts):
    """Contiguous frame chunks (one per worker / GPU)."""
    edges = np.linspace(0, n, parts + 1).astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(parts) if edges[i + 1] > edges[i]]


def run(paths, z, t, grid: Grid, workers=None, radius_px=6.0, power=4.0, smoothing_px=8,
        log=print, order=None, depth_method="sgm", agg_radius_px=24.0, agg_eps=1e-4,
        fm_radius_px=2.0, guide="C", soft_px=3.0, depth_full_fraction=0.25, depth_fm_px=6.0, sgm_p1=0.03,
        sgm_p2=0.4, sgm_edge=0.02, sgm_c_lo=0.45, sgm_c_hi=0.65, sgm_dirs=8, dark_prior=1.0,
        vol_cache=None, halo_prior=1.0, halo_radius_px=24.0, refine_r=6, refine_fm_px=1.5,
        soft_wide_px=40.0, wide_lo=0.4,
        wide_hi=0.7, tex_lo=0.025, tex_hi=0.045):
    """Full fusion. Window sizes are in FULL-RES sensor pixels (scaled to the grid
    so results don't depend on the working scale). ``order`` permutes the merge
    order of partial states (tests)."""
    radius = max(0.8, radius_px * grid.scale)
    smoothing = max(1, int(round(smoothing_px * grid.scale)))
    fill = tuple(max(2.0, f * grid.scale) for f in (8, 32, 128))
    n = len(paths)
    ow, oh = grid.size
    levels = n_levels(ow, oh)
    workers = workers or min(8, os.cpu_count() or 4)
    ctx = {"paths": [str(p) for p in paths], "z": z, "t": t, "grid": grid,
           "levels": levels, "radius": radius, "power": power}
    chunks = _chunks(n, workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
        parts = list(ex.map(_w_fold, chunks))
    idx = order if order is not None else range(len(parts))
    idx = list(idx)
    st = parts[idx[0]]
    for i in idx[1:]:
        st = merge(st, parts[i])
    log(f"  pass 1 (A, B-depth, C) folded {n} frames in {len(parts)} parts, {levels} pyramid levels")
    if depth_method == "sgm":
        g_img = collapse(st.c_coef + [(st.c_base / max(st.count, 1)).astype(np.float32)])
        guide_full = np.clip(g_img @ LUMA, 0, 1).astype(np.float32)
        ds = depth_full_fraction / grid.scale        # depth res = fixed fraction of the sensor
        ctx.update(dsize=(max(16, int(round(ow * ds))), max(16, int(round(oh * ds)))),
                   fm_radius=max(0.8, depth_fm_px * depth_full_fraction), sgm_p1=sgm_p1, sgm_p2=sgm_p2,
                   sgm_edge=sgm_edge, sgm_c_lo=sgm_c_lo, sgm_c_hi=sgm_c_hi, sgm_dirs=sgm_dirs,
                   dark_prior=dark_prior, vol_cache=vol_cache, halo_prior=halo_prior,
                   halo_radius_px=halo_radius_px, depth_full_fraction=depth_full_fraction)
        depth_s, conf = depth_sgm(ctx, chunks, workers, guide_full, log)
        ctx["soft_sigma"] = max(0.8, soft_px * grid.scale)
        ctx["soft_sigma_wide"] = max(2.0, soft_wide_px * grid.scale)
        if refine_r:
            # fine-scale refinement inside +-refine_r frames of the SGM depth,
            # only where the focus evidence is confident (texture)
            ctx.update(depth=depth_s, refine_r=refine_r,
                       refine_fm_r=max(0.6, refine_fm_px * grid.scale))
            with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
                rparts = list(ex.map(_w_refine, chunks))
            rb, ri = rparts[0][0].copy(), rparts[0][1].copy()
            for b2, i2 in rparts[1:]:
                m = _take(rb, ri, b2, i2)
                rb[m] = b2[m]
                ri[m] = i2[m]
            confs = cv2.GaussianBlur(conf, (0, 0), max(1.0, 6 * grid.scale))
            refined = np.where(np.isfinite(rb), ri.astype(np.float32), depth_s)
            depth_s = np.where(confs > 0.5, cv2.medianBlur(refined, 3), depth_s).astype(np.float32)
            log(f"  refine pass (+-{refine_r} frames, fine FM) done")
    elif depth_method == "cvf":
        if guide == "C":     # crisp edges stop aggregation crossing depth steps
            g_img = collapse(st.c_coef + [(st.c_base / max(st.count, 1)).astype(np.float32)])
        else:
            g_img = (st.a_num / np.maximum(st.a_den, 1e-30)[:, :, None]).astype(np.float32)
        ctx.update(guide=np.clip(g_img @ LUMA, 0, 1).astype(np.float32),
                   agg_r=max(2, int(round(agg_radius_px * grid.scale))), agg_eps=agg_eps,
                   fm_radius=max(0.8, fm_radius_px * grid.scale),
                   depth_blur=max(0.8, 2.0 * grid.scale))
        depth_s, conf = depth_cvf(ctx, chunks, workers, order)
        ctx["soft_sigma"] = max(0.8, soft_px * grid.scale)
        log(f"  pass 2 (cost-volume-filtered depth, r={ctx['agg_r']} px) done")
    else:
        depth_s, conf = smooth_depth_map(st, smoothing, fill_sigmas=fill)
    ctx["depth"] = depth_s
    if depth_method == "sgm":
        with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
            accs = list(ex.map(_w_b2dual, chunks))
        n1 = sum(a[0] for a in accs); d1 = sum(a[1] for a in accs)
        n2 = sum(a[2] for a in accs); d2 = sum(a[3] for a in accs)
        b1 = n1 / np.maximum(d1, 1e-12)[:, :, None]
        b2 = n2 / np.maximum(d2, 1e-12)[:, :, None]
        # decisive mix: any real texture -> narrow blend (a soft mix leaked the
        # 40 px wide blend into textured areas as haze); wide only where the
        # focus evidence is clearly absent
        # Wide blend only where BOTH signals say "nothing here": weak focus
        # evidence AND no texture in the composed image. Measured texture
        # (brightness-normalised |Laplacian| of C): background 0.015-0.021,
        # matrix 0.046-0.113, smooth crystal faces 0.012-0.03 (those keep the
        # narrow blend through their high focus confidence).
        cs = cv2.GaussianBlur(conf, (0, 0), max(2.0, 10 * grid.scale))
        dw_, dh_ = ctx["dsize"]
        gd = cv2.resize(guide_full, (dw_, dh_), interpolation=cv2.INTER_AREA)     # fixed scale
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gd, (0, 0), 0.8), cv2.CV_32F, ksize=3))
        tex = cv2.GaussianBlur(lap, (0, 0), 3.0) / (cv2.GaussianBlur(gd, (0, 0), 3.0) + 0.02)
        tex = cv2.resize(tex, (ow, oh), interpolation=cv2.INTER_LINEAR)
        c_focus = np.clip((cs - wide_lo) / (wide_hi - wide_lo), 0, 1)
        c_tex = np.clip((tex - tex_lo) / (tex_hi - tex_lo), 0, 1)
        cm = (1.0 - (1.0 - c_focus) * (1.0 - c_tex))[:, :, None]
        b_img = (cm * b1 + (1 - cm) * b2).astype(np.float32)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_w_init, initargs=(ctx,)) as ex:
            accs = list(ex.map(_w_b2, chunks))
        num = sum(a.num for a in accs)
        den = sum(a.den for a in accs)
        b_img = (num / np.maximum(den, 1e-12)[:, :, None]).astype(np.float32)
        b_img = np.where((den > 1e-6)[:, :, None], b_img, st.b_col)
    log("  B composition pass done")
    # Noise per pyramid level from two ADJACENT middle frames: their coefficient
    # difference is ~sqrt(2) x noise; the MAD ignores the few pixels whose focus
    # changed. (Frame 0's own coefficient MAD includes real detail -> sigma too
    # high -> shrinkage erased detail.)
    mid = max(0, n // 2 - 1)
    pa = lap_pyramid(load_aligned(paths[mid], z[mid], t[mid], grid), levels)
    pb = lap_pyramid(load_aligned(paths[mid + 1], z[mid + 1], t[mid + 1], grid), levels)
    st.noise_mad = [float(1.4826 * np.median(np.abs((pa[l] - pb[l]) @ LUMA)) / math.sqrt(2))
                    for l in range(levels - 1)]
    log(f"  noise sigma per level: {[round(v * 1e4, 2) for v in st.noise_mad]} (x1e-4)")
    out, maps, pyrs = finalize(st, depth_s, b_img)
    return out, {"depth": depth_s, "conf": conf, "replaced": maps, "levels": levels, "state": st,
                 "pyramids": pyrs}
