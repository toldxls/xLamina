"""Similarity transforms in centred pixel coordinates.

Conventions (used everywhere in the prototype; the CUDA port mirrors them):

* Pixel centres sit at integer coordinates. The image centre is
  ``c = ((W-1)/2, (H-1)/2)``.
* A point is the complex number ``p = (x - cx) + 1j*(y - cy)``.
* A similarity is a pair ``(z, t)``: ``z = s*exp(i*theta)``, ``t`` complex,
  acting as ``p' = z*p + t``. Arrays of transforms are parallel complex arrays.
* ``T_k`` maps REFERENCE-grid coordinates to FRAME-k coordinates. Warping
  frame k onto the reference grid samples frame k at ``T_k(p_ref)``.
* A pair measurement ``P_ij`` maps frame-i coordinates to frame-j coordinates:
  ``P_ij = T_j o T_i^-1``, so ``z_ij = z_j / z_i`` and ``t_ij = t_j - z_ij*t_i``.
  Both are LINEAR in (log z) and (t), which is what makes the global solve exact.
* Centred coordinates scale exactly with resolution when downsampling with
  pixel-area convention (cv2 INTER_AREA): at work = f * full,
  ``p_full = p_work / f``, so ``t_full = t_work / f`` and z is unchanged.
"""
from __future__ import annotations

import numpy as np


def centre(width: int, height: int) -> complex:
    return complex((width - 1) / 2.0, (height - 1) / 2.0)


def compose(a_z, a_t, b_z, b_t):
    """(a o b)(p) = a(b(p))."""
    return a_z * b_z, a_z * b_t + a_t


def invert(z, t):
    return 1.0 / z, -t / z


def apply(z, t, p):
    return z * p + t


def rebase(z, t, ref: int):
    """Re-express transforms relative to frame ``ref``: T'_k = T_k o T_ref^-1."""
    iz, it = invert(z[ref], t[ref])
    return compose(z, t, iz, it)


def to_affine(z: complex, t: complex, width: int, height: int) -> np.ndarray:
    """2x3 matrix M in plain pixel coords: [x', y'] = M @ [x, y, 1].

    Feed to ``cv2.warpAffine(src, M, size, flags=... | cv2.WARP_INVERSE_MAP)``
    to sample ``src`` at T(dst) -- i.e. warp a frame onto the reference grid.
    """
    c = centre(width, height)
    zr, zi = z.real, z.imag
    ox = c.real + t.real - zr * c.real + zi * c.imag
    oy = c.imag + t.imag - zi * c.real - zr * c.imag
    return np.array([[zr, -zi, ox], [zi, zr, oy]], dtype=np.float64)


def scale_of(z):
    return np.abs(z)


def theta_of(z):
    return np.angle(z)
