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

def finalize(st: State, depth_s, b_img, agree=3.0, dominance=4.0, noise_k=3.0):
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
    # hybrid: luma from CBn, chroma from B (BT.709 on the encoded values)
    out["H"] = _swap_chroma(out["CBn"], b_img)
    return out, maps


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


def _chunks(n, parts):
    """Contiguous frame chunks (one per worker / GPU)."""
    edges = np.linspace(0, n, parts + 1).astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(parts) if edges[i + 1] > edges[i]]


def run(paths, z, t, grid: Grid, workers=None, radius_px=6.0, power=4.0, smoothing_px=8,
        log=print, order=None, depth_method="cvf", agg_radius_px=24.0, agg_eps=1e-4,
        fm_radius_px=2.0, guide="C", soft_px=3.0):
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
    if depth_method == "cvf":
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
    out, maps = finalize(st, depth_s, b_img)
    return out, {"depth": depth_s, "conf": conf, "replaced": maps, "levels": levels, "state": st}
