"""Transform conventions, pair alignment, the exact global solve, geometry."""
import cv2
import numpy as np
import pytest

from stackproto import align, geometry, sim
from stackproto.synth import make_texture


def test_compose_invert_rebase_affine_roundtrip():
    z, t = 1.001 * np.exp(1j * 0.002), 3.5 - 2.25j
    iz, it = sim.invert(z, t)
    cz, ct = sim.compose(z, t, iz, it)
    assert abs(cz - 1) < 1e-12 and abs(ct) < 1e-12
    W, H = 640, 480
    M = sim.to_affine(z, t, W, H)
    x, y = 100.0, 50.0
    c = sim.centre(W, H)
    p2 = z * (complex(x, y) - c) + t + c
    got = M @ np.array([x, y, 1.0])
    assert np.allclose(got, [p2.real, p2.imag])
    zs = np.array([1.0, 1.001, 0.999]) + 0j
    ts = np.array([0, 1 + 1j, -2j])
    rz, rt = sim.rebase(zs, ts, 1)
    assert abs(rz[1] - 1) < 1e-12 and abs(rt[1]) < 1e-12


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(3)
    from stackproto.frames import luma
    return luma(make_texture(rng, 400, 600))


@pytest.mark.parametrize("z,t", [
    (1.0 + 0j, 3.3 - 1.7j),
    (1.0015 * np.exp(1j * np.deg2rad(0.05)), -12.25 + 4.5j),
    (0.9993 + 0j, 0.4 + 0.2j),
])
def test_pair_alignment_recovers_known_warp(texture, z, t):
    I0 = texture
    h, w = I0.shape
    from scipy import ndimage
    iz, it = sim.invert(z, t)
    M = sim.to_affine(iz, it, w, h)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    X = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
    Y = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
    I1 = ndimage.map_coordinates(I0, [Y, X], order=3, mode="mirror").astype(np.float32)
    p0, f = align.pyramid(I0, 100)
    p1, _ = align.pyramid(I1, 100)
    r = align.align_pair(p0, p1, f)
    assert abs(abs(r.z) - abs(z)) < 3e-6
    assert abs(np.angle(r.z) - np.angle(z)) < np.deg2rad(5e-4)
    assert abs(r.t - t) < 2e-3
    assert r.ok


def test_global_solve_is_exact_for_consistent_pairs():
    rng = np.random.default_rng(0)
    n = 25
    zt = np.exp(np.cumsum(rng.normal(0, 3e-4, n)) + 1j * np.cumsum(rng.normal(0, 2e-4, n)))
    tt = np.cumsum(rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n))
    zt, tt = sim.rebase(zt, tt, 0)
    pairs = []
    for i, j in align.make_pairs(n, (1, 2, 4, 8)):
        zij = zt[j] / zt[i]
        pairs.append(align.PairResult(i, j, zij, tt[j] - zij * tt[i], 1e-12, 1e-12, 1e-4, 1e-4,
                                      1.0, 1.0, 10000, 5, True))
    z, t, info = align.solve_global(n, pairs)
    assert np.max(np.abs(z - zt)) < 1e-10
    assert np.max(np.abs(t - tt)) < 1e-8


def test_geometry_crop_for_pure_translation():
    W, H = 1000, 800
    z = np.ones(3, complex)
    t = np.array([0, 10 + 0j, -4 - 6j])        # ref -> frame coords
    x0, y0, x1, y1 = geometry.common_rect(z, t, W, H, margin=0)
    # frame 1 content shifted: ref pixel x maps to x+10 -> ref x <= W-1-10 valid
    assert (x0, y0, x1, y1) == (4, 6, W - 1 - 10, H - 1)
    rep = geometry.report(z, t, W, H, um_per_px=0.5, margin=0)
    nc = rep["policies"]["native-crop"]
    assert nc["crop_px"] == {"left": 4, "right": 10, "top": 6, "bottom": 0}
    rs = rep["policies"]["rescaled"]
    assert rs["width"] == W
    assert abs(rs["um_per_px"] - 0.5 * nc["width"] / W) < 1e-12
    assert rep["policies"]["native-full"]["min_frames_covering"] < 3


def test_reference_choice_zoom_semantics():
    # s_k > 1 relative to frame 0 = frame k content larger (more zoomed in)
    z0 = np.array([1.0, 1.001, 1.002]) + 0j
    t0 = np.zeros(3, complex)
    assert geometry.choose_reference(z0, t0, "most-zoomed") == 2
    zr, tr = sim.rebase(z0, t0, 2)
    W, H = 1000, 800
    x0, y0, x1, y1 = geometry.common_rect(zr, tr, W, H, margin=0)
    assert (x0, y0, x1, y1) == (0, 0, W - 1, H - 1)    # nothing shrinks -> no crop
