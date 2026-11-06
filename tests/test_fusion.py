"""A8: fusion folds are order-independent (the multi-GPU invariant)."""
import numpy as np
import pytest

from stackproto import fusion, sim
from stackproto.frames import list_frames
from stackproto.synth import SynthParams, generate, load_truth


@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    p = SynthParams(width=320, height=220, n_frames=14, step_um=8, relief="steps", noise=0.004)
    return generate(tmp_path_factory.mktemp("fus") / "steps", p, verbose=False)


def _run(folder, workers, order=None):
    _, z, t = load_truth(folder)
    z, t = sim.rebase(z, t, 0)
    W, H = 320, 220
    grid = fusion.Grid(W, H, (0, 0, W - 1, H - 1), scale=1.0)
    out, _ = fusion.run(list_frames(folder), z, t, grid, workers=workers, order=order,
                        log=lambda *_: None)
    return out


def test_split_and_merge_order_do_not_change_the_result(steps):
    one = _run(steps, 1)
    three = _run(steps, 3, order=[2, 0, 1])
    for k in ("B", "B0", "C", "CB", "Cn", "CBn", "H"):
        assert np.array_equal(one[k], three[k]), k
    # A sums in a different order: equal to float rounding
    assert np.max(np.abs(one["A"] - three["A"])) < 1e-6


def test_bc_slider_endpoints(steps):
    _, z, t = load_truth(steps)
    z, t = sim.rebase(z, t, 0)
    grid = fusion.Grid(320, 220, (0, 0, 319, 219), scale=1.0)
    out, info = fusion.run(list_frames(steps), z, t, grid, workers=2, log=lambda *_: None)
    pb, pc = info["pyramids"]["B"], info["pyramids"]["Cn"]
    assert np.allclose(fusion.blend_bc(pb, pc, 0.0), out["B"], atol=1e-5)
    assert np.allclose(fusion.blend_bc(pb, pc, 1.0), out["Cn"], atol=1e-5)
