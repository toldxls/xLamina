# Prototype findings: alignment, geometry and optical-train diagnostics

*Phase A (A1–A6), 2026-11-05, all on synthetic stacks with known ground truth
(`stackproto synth` / `validate`). Real-stack validation is still pending.*

## What works (validated)

On a **flat target**, at full size (1740 × 1160, 120 frames, 2 µm steps, `faulty` preset):

| Quantity | Injected | Measured |
|---|---|---|
| scale rate ε | 0.0100 /mm | 0.00976 ± 0.00004 /mm (−2.4 %) |
| rail tilt x / y | 0.25° / −0.12° | 0.216 ± 0.017° / −0.123 ± 0.008° |
| zoom centre (FOE) | (−342, 212) px | (−329, 208) px |
| lead-screw periodic error | 60 µm, 0.296 µm | 60 µm, 0.297 µm |
| jitter RMS x / y | 0.20 / 0.09 px | 0.23 / 0.10 px |
| native crop | (3, 4, 1735, 1156) | identical |
| per-frame corner error | — | 0.33 px RMS, 0.52 px max |

Analysis takes ~25 s for 120 frames on an M3 Pro (align scale 0.5, baselines 1/2/4/8).

## 1. OpenCV warps are too coarse for scale measurement

`cv2.warpAffine`/`remap` snap sample positions to **1/32 px**. A per-frame scale
change of ~160 ppm moves pixels only ~0.08 px, so the snapping biased individual
pair scales by up to ~20 % (pair (0,1) exact, pair (10,11) −22 %).

- Fix: OpenCV gets close (coarse levels), then 4 Gauss–Newton iterations with
  exact cubic B-spline interpolation (`scipy.ndimage.map_coordinates`).
- The synthetic generator must warp exactly too, or the ground truth is wrong.
- **CUDA consequence:** texture-unit interpolation uses 1/256 px weights. The
  alignment kernels must interpolate in software (FP32 weights). Hardware
  textures are fine for the fusion warp.

## 2. The global solve is exact and linear

With `P_ij = T_j ∘ T_i⁻¹` in centred complex coordinates:
- `log z_ij = log z_j − log z_i` (linear in log z)
- `t_ij = t_j − z_ij t_i` (linear in t once the z's are known)

Two weighted least-squares solves, with Huber IRLS on multi-baseline pairs
(1, 2, 4, 8), recover the transforms exactly from consistent pairs
(`test_global_solve_is_exact_for_consistent_pairs`). Multi-baseline pairs stop
the random-walk drift of a plain neighbour-to-neighbour chain.

## 3. KEY FINDING: specimen relief biases scale and drift

On a 3D specimen, intensity-based alignment reads part of the specimen's
**shape** as magnification or drift.

| Synthetic specimen | ε true → measured | rail tilt x true → measured |
|---|---|---|
| flat target | 0.020 → 0.0203 ✅ | 0.30° → 0.31° ✅ |
| tilted flat target | 0.020 → 0.0183 | 0.30° → **0.65°** ❌ |
| dome (insect-like) | 0.020 → **0.0085** ❌ | 0.30° → 0.54° ❌ |

A controlled pair experiment, rendering frames with swapped blur and geometry:
- same blur, different geometry → +161.6 ppm (true 160): the geometry measurement is exact;
- different blur, same geometry → **−95.8 ppm**: the bias comes purely from the blur change.

Scaling:
- The bias scales with **NA²**: −93 ppm at NA 0.28 vs −20.5 ppm at NA 0.14 (4.5×, predicted 4×).
- It also scales with step size and surface slope.

Rough real-world estimate (10× NA 0.28, ~0.34 µm/px, 2 µm steps, insect-like
slopes): the bias is ~tens of ppm per frame, **much larger than the
telecentricity residual we want to measure** (~0.5 ppm per frame for a pupil 2 m
away).

Tried and **failed**:
- depth-gated alignment: weight each pair to pixels at the midpoint depth;
- per-pixel blur equalisation: re-blur the sharper frame using the depth map, with true depth and true blur rate;
- high-pass filtering: made it worse.

Mechanism: defocused light spreads in from *neighbouring* regions at other
depths. Its pattern changes asymmetrically between frames on a curved or sloped
surface, and no per-pixel model captures it. Real optics do this too (out-of-focus
light adds), so this is not a generator artefact. It survived after the generator
was fixed (see 5).

Implications:
1. **Optics diagnostics need a flat target.** Specimen stacks give a report
   flagged with a relief caveat (`--target specimen`, the default).
2. **Tilted targets bias the tilt reading.** Proposed cure, the **reversal
   protocol** (not yet tested): shoot the flat target, rotate it 180° about the
   optical axis, shoot again, and average. Target slope flips sign; rail tilt
   doesn't.
3. **Stacking with telecentric optics should probably lock scale**
   (`--lock-scale`). The estimate is dominated by relief bias, not by real
   magnification change. Check on real stacks.
4. Real stackers (Helicon/Zerene) presumably carry the same bias in their scale
   correction. Untested.

## 4. The zoom centre is the FOE, not the optical axis

A camera translating along a tilted rail zooms about the **focus of expansion**,
not the optical axis. The two coincide only if the rail is parallel to the axis.

- The FOE is only defined when ε is clearly non-zero; near-telecentric systems leave it undefined.
- The rail-tilt reading depends on where the axis is assumed to be, with sensitivity ε × axis error per mm. Supply the lens-centering result with `--axis-pct`.
- The design doc's earlier claim that the "scale fixed point ≈ optical axis" was wrong. Corrected.

## 5. Synthetic-generator pitfalls (fixed)

- **Canvas margin must exceed 3σ of the largest defocus blur.** Mirror padding
  inside the frame produced a fake ±30 ppm/pair bias that flipped sign with the
  direction of blur change (−30 sharpening / +30 blurring). Per-frame scale
  error was 642 ppm RMS before the fix and 37 after.
- **Defocus is now exact**: a frequency-domain Gaussian per layer, one inverse FFT per frame.

## Open items

- **Residual flat-target roll bias**: a smooth hump of ±17 mdeg RMS per frame on far-defocused frames, antisymmetric about focus (the fitted rate is fine: 52 vs 50 mdeg/mm). Also ε reads −2.4 % (≈ 6σ). Cause unknown.
- Validate everything on **real** stacks:
  1. a flat-target stack (the diagnostic);
  2. a specimen stack (how big is the relief bias in reality?);
  3. the same specimen with scale locked vs free.
- Test the reversal protocol.
