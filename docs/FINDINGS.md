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

## 6. First real stack (2026-11-05): mineral specimen, 163 frames

- Setup: Canon EOS M6 II, 6960 × 4640, 16-bit Adobe RGB TIFF, 0.468 µm steps
  (76 µm of travel), 0.98 mm FOV (0.1408 µm/px, ~23×, presumably 20× NA ~0.42).
  Higher frame numbers move the camera **away** from the subject.
- Alignment: 637/637 pairs usable, NCC ≥ 0.992. Runtime 3 min on the Mac
  (luma at 1740 × 1160).
- Measured: scale **+2.5 % across the stack** (+153 ppm/frame, growing as the
  camera moves away), "rail tilt" 7° x / 18° y, y drift ~175 px, roll −0.49°.
- **The scale ramp has the wrong sign for perspective.** Moving away should
  shrink the image with a normal (entocentric) lens; it grew. Real perspective
  would need a pupil 3 mm away, on the far side of the specimen. The size fits
  the specimen-relief bias (NA² is ~2× the synthetic case; steep crystal relief).
  **Probably not real geometry.** Needs a flat-target stack with the same setup
  to confirm.
- **Helicon applied the same ~2.5 % ramp**, referenced to the FIRST frame (its
  B output's scale matches frame 0 to 0.007 %).
  - Frame 0 is the least zoomed-in frame here, the reference choice with the
    largest crop: 6960 × 4640 → 6714 × 4404. The 246 px width loss needs the
    scale term; drift alone is ~60 px.
  - Helicon output µm/px over depth: **0.1408 (first frame's depth) to 0.1443
    (last frame's depth)**, 0.1424 mid. AutoFOV's single calibrated number is
    right only for the deepest content.
- Our native-crop with the middle reference: 6840 × 4429 (L29 R91 T88 B123).
- Command: `stackproto compare <analysis_dir> <helicon.tif> ...`.

## 7. Fusion prototypes (A8), 2026-11-05

All folds are order-independent. 1 worker vs 3 workers with a scrambled merge
order gives bit-identical B / C / variants (`test_fusion.py`): the multi-GPU
invariant holds.

**Synthetic "steps" stack** (crystal-like plateaus; PSNR vs ground truth, higher = better):

| Method | All | Halo zone | Plateaus |
|---|---|---|---|
| A weighted avg | 34.9 | 31.1 | 39.6 |
| B depth map | 33.5 | 29.0 | 44.3 |
| C pyramid | 37.4 | **34.0** | 41.2 |
| **Cn** noise-aware pyramid | **37.6** | 33.5 | **44.3** |
| CB depth-guided | 37.5 | 33.3 | 44.2 |
| H luma CBn / chroma B | 36.8 | 32.5 | 44.3 |

**Real stack** (163 frames, compared at half and full res against Helicon B, C
and the user's B+C composite; montages: `~/Desktop/stacker_vs_helicon*.jpg`):
- **C ≈ Helicon C** in detail. It is noisier in dark low-signal gaps and shows
  seams there; Helicon evidently suppresses that.
- **Cn is the best of ours**: clean at half res, needles intact, close to the
  user's composite. At full res it picks up some of B's glow in dark gaps.
- **B is NOT at Helicon-B quality.** The depth-map confidence is the open
  problem; each fix helped one scale/region and hurt another:
  1. relative peak (best−mean)/best → blotchy patchworks in defocused background;
  2. guided by the B0 argmax colour → kept those patch outlines (fixed: guide with A);
  3. global noise-floor test → clean background at half res, but at full res
     fine low-contrast detail sits near the per-pixel noise floor, so the fan
     centre lost its needles;
  4. per-pixel peak/mean ratio (current) → full-res fan fixed, half-res
     background patch outlines are back, and dark gaps glow at full res.

  Needs a real depth estimator: multi-scale focus measure, regularised
  (MRF/graph-cut style) labelling with occlusion awareness, not thresholds.
- Rejected: **Cs** (global soft-threshold shrinkage + selected base). Fine detail
  at full res is noise-sized, so it erased the needles. Synthetic 34.5 vs C 37.4.
- Noise per pyramid level is now estimated from two ADJACENT frames' coefficient
  difference (MAD/√2). Frame 0's own MAD includes real detail.
- Window sizes are in full-res sensor px and scaled to the working grid, so
  half- and full-res runs use the same physical windows.
- Speed (CPU, Mac, 6 workers): all methods in ONE shared read pass (broadcast
  mode), 163 frames → 3420 × 2214 in ~95 s. Helicon on a 3080 Ti: B ~20 s,
  C ~9–10 s at full res.

## Open items

- **Residual flat-target roll bias**: a smooth hump of ±17 mdeg RMS per frame on far-defocused frames, antisymmetric about focus (the fitted rate is fine: 52 vs 50 mdeg/mm). Also ε reads −2.4 % (≈ 6σ). Cause unknown.
- Validate everything on **real** stacks:
  1. a flat-target stack (the diagnostic);
  2. a specimen stack (how big is the relief bias in reality?);
  3. the same specimen with scale locked vs free.
- Test the reversal protocol.
- **B depth-map estimator** (see 7). Cn depends on B in low-contrast areas.
