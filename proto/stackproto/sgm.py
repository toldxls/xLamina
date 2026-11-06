"""Semi-global matching over a focus cost volume (depth-from-focus regularisation).

Cost volume C[y, x, k] in [0, 1] (0 = sharpest). SGM approximates the MRF
    E(D) = sum_p C(p, D_p) + sum_{p~q} [P1 if |D_p - D_q| = 1, P2' if > 1]
by dynamic programming along scanlines (Hirschmueller 2008):
    L_r(p, d) = C(p, d) + min(L_r(p-r, d), L_r(p-r, d +- 1) + P1,
                              min_k L_r(p-r, k) + P2'(p)) - min_k L_r(p-r, k)
P2' shrinks across image edges so depth may jump there. 4 or 8 directions.
"""
from __future__ import annotations

import numpy as np


def _pass(C, P1, P2, axis, reverse, S):
    """One scanline direction along ``axis`` (1 = x, 0 = y); accumulates into S."""
    n = C.shape[axis]
    rng = range(n - 1, -1, -1) if reverse else range(n)
    prev = None
    for i in rng:
        c = C[:, i, :] if axis == 1 else C[i, :, :]
        p2 = P2[:, i] if axis == 1 else P2[i, :]
        if prev is None:
            cur = c.astype(np.float32).copy()
        else:
            pm = prev.min(axis=1, keepdims=True)
            up = np.empty_like(prev)
            up[:, 1:] = prev[:, :-1]
            up[:, 0] = np.inf
            dn = np.empty_like(prev)
            dn[:, :-1] = prev[:, 1:]
            dn[:, -1] = np.inf
            m = np.minimum(np.minimum(prev, np.minimum(up, dn) + P1), pm + p2[:, None])
            cur = c + m - pm
        if axis == 1:
            S[:, i, :] += cur
        else:
            S[i, :, :] += cur
        prev = cur


def _diag_pass(C, P1, P2, dy, dx, S):
    """Diagonal direction (dy, dx in {-1, 1}) via per-diagonal sweeps."""
    H, W, K = C.shape
    ys = range(H) if dy > 0 else range(H - 1, -1, -1)
    prev = np.full((W, K), np.nan, np.float32)    # indexed by x of the previous row
    for y in ys:
        c = C[y].astype(np.float32)
        if np.isnan(prev[0, 0]) and np.isnan(prev).all():
            cur = c.copy()
        else:
            # predecessor of (y, x) is (y - dy, x - dx)
            pr = np.empty_like(prev)
            if dx > 0:
                pr[1:] = prev[:-1]
                pr[0] = np.nan
            else:
                pr[:-1] = prev[1:]
                pr[-1] = np.nan
            valid = ~np.isnan(pr[:, 0])
            cur = c.copy()
            if valid.any():
                q = pr[valid]
                pm = q.min(axis=1, keepdims=True)
                up = np.empty_like(q); up[:, 1:] = q[:, :-1]; up[:, 0] = np.inf
                dn = np.empty_like(q); dn[:, :-1] = q[:, 1:]; dn[:, -1] = np.inf
                m = np.minimum(np.minimum(q, np.minimum(up, dn) + P1), pm + P2[y, valid][:, None])
                cur[valid] = c[valid] + m - pm
        S[y] += cur
        prev = cur


def sgm(C, P1, P2, directions=8):
    """Aggregate cost volume C (H, W, K) float32; P2 is an (H, W) map."""
    S = np.zeros(C.shape, np.float32)
    for axis, rev in ((1, False), (1, True), (0, False), (0, True)):
        _pass(C, P1, P2, axis, rev, S)
    if directions == 8:
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            _diag_pass(C, P1, P2, dy, dx, S)
    return S


def argmin_subpixel(S):
    k = np.argmin(S, axis=2)
    K = S.shape[2]
    kc = np.clip(k, 1, K - 2)
    yy, xx = np.indices(k.shape)
    a, b, c = S[yy, xx, kc - 1], S[yy, xx, kc], S[yy, xx, kc + 1]
    den = a - 2 * b + c
    off = np.where(den > 1e-9, 0.5 * (a - c) / np.where(den > 1e-9, den, 1), 0.0)
    off = np.where((k >= 1) & (k <= K - 2), np.clip(off, -0.5, 0.5), 0.0)
    return (kc + off).astype(np.float32) if True else k
