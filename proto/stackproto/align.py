"""Pairwise similarity alignment and the multi-baseline global solve.

Pair (i, j): find P with  g * I_j(P(p)) + beta ~= I_i(p),   P(p) = z*p + t
(centred coords, see sim.py), i.e. P maps frame-i coords to frame-j coords.

* coarse: phase correlation at the coarsest pyramid level (translation only);
* fine: Gauss-Newton on (a, b, tx, ty, g, beta) with z = 1 + a + i b, using the
  ESM mean-gradient Jacobian, coarse-to-fine; cubic interpolation;
* every pair returns its parameter covariance (sigma^2 (J^T J)^-1), used to
  weight the global solve.

Global solve (exact, linear -- see sim.py): log z_j - log z_i = log z_ij, then
t_j - z_ij t_i = t_ij, weighted least squares with IRLS outlier rejection.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from . import sim


# ----------------------------------------------------------------------------- pyramids

def pyramid(img: np.ndarray, min_width: int = 200):
    """Level 0 = img. Returns (levels, factors) with factors[l] = (fx, fy) vs level 0."""
    levels, factors = [np.ascontiguousarray(img, dtype=np.float32)], [(1.0, 1.0)]
    while levels[-1].shape[1] // 2 >= min_width:
        prev = levels[-1]
        h, w = prev.shape
        sm = cv2.GaussianBlur(prev, (0, 0), 0.8)
        nxt = cv2.resize(sm, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        levels.append(nxt)
        factors.append((nxt.shape[1] / img.shape[1], nxt.shape[0] / img.shape[0]))
    return levels, factors


def _grad(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1, scale=0.5, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1, scale=0.5, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


@dataclass
class PairResult:
    i: int
    j: int
    z: complex
    t: complex
    var_a: float        # variance of Re(z) ~ ln s
    var_b: float        # variance of Im(z) ~ theta
    var_tx: float
    var_ty: float
    gain: float
    ncc: float
    n_px: int
    iters: int
    ok: bool


def _phase_init(I0, I1) -> complex:
    win = cv2.createHanningWindow(I0.shape[::-1], cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(I0.astype(np.float32), I1.astype(np.float32), win)
    return complex(dx, dy)


def _warp_cv2(I1):
    """Fast warp. OpenCV snaps sample positions to 1/32 px -- fine for getting
    close, NOT for the final estimate (it biased per-frame scale by ~20%)."""
    h, w = I1.shape
    return lambda M, X, Y: cv2.warpAffine(I1, M, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_REPLICATE)


def _warp_exact(I1):
    """Cubic B-spline at exact float positions (the CUDA port must also
    interpolate in software: texture hardware has 1/256 px weights)."""
    coef = ndimage.spline_filter(I1, 3, mode="mirror", output=np.float32)
    return lambda M, X, Y: ndimage.map_coordinates(coef, [Y, X], order=3, prefilter=False,
                                                   mode="mirror", output=np.float32)


def _gn_level(I0, gx0, gy0, warp, p, max_iter, tol=2e-4, wts=None):
    """Gauss-Newton at one level. p = [a, b, tx, ty, g, beta] (level units).

    ESM-style mean gradient: (grad I0 + grad(I1 o W)) / 2; the gradient of the
    warped image stands in for the warped gradient (they differ by z ~ 1+1e-3,
    which changes convergence speed, not the fixed point).
    """
    h, w = I0.shape
    c = sim.centre(w, h)
    R = math.hypot(w, h) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dxs, dys = ((xs - c.real) / R).ravel(), ((ys - c.imag) / R).ravel()
    I0f, gx0f, gy0f = I0.ravel(), gx0.ravel(), gy0.ravel()
    wf = None if wts is None else wts.ravel()
    it = 0
    JtJ = r = sel = I1w = None
    for it in range(1, max_iter + 1):
        a, b, tx, ty, g, beta = p
        M = sim.to_affine(complex(1 + a, b), complex(tx, ty), w, h)
        X = (M[0, 0] * xs + M[0, 1] * ys + M[0, 2]).astype(np.float32)
        Y = (M[1, 0] * xs + M[1, 1] * ys + M[1, 2]).astype(np.float32)
        sel = ((X >= 3) & (X <= w - 4) & (Y >= 3) & (Y <= h - 4)).ravel()
        I1w2 = warp(M, X, Y)
        gx1w, gy1w = _grad(I1w2)
        I1w = I1w2.ravel()
        Gx = 0.5 * (g * gx1w.ravel() + gx0f)[sel]
        Gy = 0.5 * (g * gy1w.ravel() + gy0f)[sel]
        dx, dy = dxs[sel], dys[sel]
        Iw = I1w[sel]
        J = np.empty((Gx.size, 6), np.float32)
        J[:, 0] = Gx * dx + Gy * dy
        J[:, 1] = -Gx * dy + Gy * dx
        J[:, 2] = Gx
        J[:, 3] = Gy
        J[:, 4] = Iw
        J[:, 5] = 1.0
        r = (g * Iw + beta - I0f[sel]).astype(np.float32)
        if wf is not None:
            sw = np.sqrt(wf[sel])
            J *= sw[:, None]
            r = r * sw
        JtJ = (J.T @ J).astype(np.float64)
        Jtr = (J.T @ r).astype(np.float64)
        try:
            d = -np.linalg.solve(JtJ, Jtr)
        except np.linalg.LinAlgError:
            break
        p = p + np.array([d[0] / R, d[1] / R, d[2], d[3], d[4], d[5]])
        if max(abs(d[0]), abs(d[1]), abs(d[2]), abs(d[3])) < tol:
            break
    return p, it, JtJ, r, sel, I1w, R


def align_pair(pyr_i, pyr_j, factors, i=0, j=1, max_iter=(40, 25, 15, 12),
               exact_iters=4) -> PairResult:
    nl = len(pyr_i)
    top = nl - 1
    t0 = _phase_init(pyr_i[top], pyr_j[top])
    p = np.array([0.0, 0.0, t0.real, t0.imag, 1.0, 0.0])
    it_total = 0
    for lvl in range(top, -1, -1):
        if lvl != top:      # rescale translation from level lvl+1 to lvl
            fx0, fy0 = factors[lvl + 1]
            fx1, fy1 = factors[lvl]
            p[2] *= fx1 / fx0
            p[3] *= fy1 / fy0
        I0, I1 = pyr_i[lvl], pyr_j[lvl]
        gx0, gy0 = _grad(I0)
        mi = max_iter[min(top - lvl, len(max_iter) - 1)]
        p, it, JtJ, r, sel, I1w, R = _gn_level(I0, gx0, gy0, _warp_cv2(I1), p, mi, tol=1e-3)
        it_total += it
        if lvl == 0 and exact_iters:
            p, it, JtJ, r, sel, I1w, R = _gn_level(I0, gx0, gy0, _warp_exact(I1), p,
                                                   exact_iters, tol=1e-5)
            it_total += it

    n = int(sel.sum())
    s2 = float(r @ r) / max(n - 6, 1)
    try:
        cov = s2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.inf)
    a0 = pyr_i[0].ravel()[sel]
    a1 = I1w[sel]
    ncc = float(np.corrcoef(a0, a1)[0, 1]) if n > 10 else 0.0
    ok = bool(np.isfinite(p).all() and ncc > 0.3 and n > 0.3 * pyr_i[0].size)
    return PairResult(i, j, complex(1 + p[0], p[1]), complex(p[2], p[3]),
                      float(cov[0, 0] / R ** 2), float(cov[1, 1] / R ** 2),
                      float(cov[2, 2]), float(cov[3, 3]), float(p[4]), ncc, n, it_total, ok)


# ----------------------------------------------------------------------------- pair scheduling

_STACK = None


def _init_worker(path):
    global _STACK
    _STACK = np.load(path, mmap_mode="r")


def _align_chunk(pairs):
    cache = {}

    def pyr(k):
        if k not in cache:
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
            cache[k] = pyramid(np.asarray(_STACK[k]))
        return cache[k]

    out = []
    for i, j in pairs:
        (li, fac), (lj, _) = pyr(i), pyr(j)
        out.append(align_pair(li, lj, fac, i, j))
    return out


def make_pairs(n: int, baselines=(1, 2, 4, 8)):
    return [(i, i + d) for d in baselines for i in range(n - d)]


def align_stack(stack_path: str, n: int, baselines=(1, 2, 4, 8), workers=None, progress=None):
    """Align all pairs of the (N, h, w) luma stack saved at ``stack_path`` (.npy)."""
    pairs = sorted(make_pairs(n, baselines))
    chunk = max(1, len(pairs) // ((workers or os.cpu_count() or 4) * 4))
    chunks = [pairs[k:k + chunk] for k in range(0, len(pairs), chunk)]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(stack_path,)) as ex:
        for res in ex.map(_align_chunk, chunks):
            results.extend(res)
            if progress:
                progress(len(results), len(pairs))
    return results


# ----------------------------------------------------------------------------- global solve

def _wlstsq(n, rows, rhs, w, fixed):
    """Weighted complex LS. rows: list of [(col, coef), ...]; column ``fixed`` held at 0."""
    cols = [k for k in range(n) if k != fixed]
    idx = {k: m for m, k in enumerate(cols)}
    A = np.zeros((len(rows), n - 1), complex)
    for e, row in enumerate(rows):
        for col, coef in row:
            if col != fixed:
                A[e, idx[col]] += coef
    sw = np.sqrt(w)
    x, *_ = np.linalg.lstsq(A * sw[:, None], rhs * sw, rcond=None)
    full = np.zeros(n, complex)
    full[cols] = x
    return full, A


def solve_global(n: int, pairs: list[PairResult], irls_rounds=3, huber=3.0):
    """Per-frame (z, t) relative to frame 0 from pair measurements.

    Returns (z, t, info) with info['inlier'] (per pair) and per-pair residuals.
    """
    good = [p for p in pairs if p.ok]
    if not good:
        raise RuntimeError("no usable pairs")
    lz = np.array([np.log(p.z) for p in good])
    w_l = np.array([1.0 / max(0.5 * (p.var_a + p.var_b), 1e-18) for p in good])
    w_t = np.array([1.0 / max(0.5 * (p.var_tx + p.var_ty), 1e-12) for p in good])
    rows_l = [[(p.j, 1.0), (p.i, -1.0)] for p in good]
    rob_l = np.ones(len(good))
    rob_t = np.ones(len(good))
    for _ in range(irls_rounds + 1):
        L, _A = _wlstsq(n, rows_l, lz, w_l * rob_l, 0)
        res_l = lz - np.array([L[p.j] - L[p.i] for p in good])
        zf = np.exp(L)
        zij = np.array([zf[p.j] / zf[p.i] for p in good])
        rows_t = [[(p.j, 1.0), (p.i, -zij[e])] for e, p in enumerate(good)]
        tt = np.array([p.t for p in good])
        T, _A = _wlstsq(n, rows_t, tt, w_t * rob_t, 0)
        res_t = tt - np.array([T[p.j] - zij[e] * T[p.i] for e, p in enumerate(good)])
        # Huber reweighting on normalised residuals (scale by robust sigma)
        nl = np.abs(res_l) * np.sqrt(w_l)
        nt = np.abs(res_t) * np.sqrt(w_t)
        sl = 1.4826 * np.median(nl) + 1e-12
        st = 1.4826 * np.median(nt) + 1e-12
        rob_l = np.minimum(1.0, huber * sl / np.maximum(nl, 1e-12))
        rob_t = np.minimum(1.0, huber * st / np.maximum(nt, 1e-12))
    inlier = (nl < 5 * sl) & (nt < 5 * st)
    info = {
        "pairs_used": len(good), "pairs_total": len(pairs),
        "inlier": inlier, "res_logz": res_l, "res_t": res_t,
        "sigma_scale_l": sl, "sigma_scale_t": st, "good": good,
    }
    return np.exp(L), T, info
