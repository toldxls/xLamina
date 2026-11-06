"""A6 acceptance: recover injected optical-train faults from synthetic stacks."""
import numpy as np
import pytest

from stackproto import pipeline
from stackproto.synth import SynthParams, generate

FAULTS = dict(width=870, height=580, n_frames=30, step_um=8, eps_per_mm=0.02,
              axis_px=(-16.3, 0.0), tilt_deg=(0.3, 0.1), roll_deg_per_mm=0.05, noise=0.003)


def _run(tmp, relief):
    folder = generate(tmp / relief, SynthParams(relief=relief, **FAULTS), verbose=False)
    rep, A = pipeline.analyze_folder(folder, FAULTS["step_um"], fov_mm=2.0, align_scale=1.0,
                                     axis_px=FAULTS["axis_px"],
                                     out_dir=tmp / f"{relief}_out", target="flat",
                                     log=lambda *_: None)
    return pipeline.validate_synth(folder, rep, A)


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("synth"), "flat")


def test_flat_target_scale_rate(flat):
    assert abs(flat["eps_meas"] - flat["eps_true"]) < 0.05 * abs(flat["eps_true"])


def test_flat_target_tilt(flat):
    # within 3 sigma of the reported uncertainty (+0.01 deg floor)
    err = np.abs(np.array(flat["tilt_meas_deg"]) - np.array(flat["tilt_true_deg"]))
    assert np.all(err < 3 * np.array(flat["tilt_se_deg"]) + 0.01), (err, flat["tilt_se_deg"])


def test_flat_target_crop(flat):
    assert np.max(np.abs(np.array(flat["crop_meas"]) - np.array(flat["crop_true"]))) <= 1


def test_flat_target_corner_error_subpixel(flat):
    # dominated by the end frames, which are far out of focus on a flat target
    assert flat["corner_err_px_rms"] < 0.4


@pytest.mark.xfail(strict=True, reason="KNOWN LIMITATION (docs/FINDINGS.md): on a 3D specimen, "
                   "defocused light from other depths biases intensity alignment's scale by ~-50% "
                   "here. If this starts passing, the bias has been fixed -- update FINDINGS.md.")
def test_dome_specimen_scale_rate(tmp_path_factory):
    d = _run(tmp_path_factory.mktemp("synth_dome"), "dome")
    assert abs(d["eps_meas"] - d["eps_true"]) < 0.05 * abs(d["eps_true"])
