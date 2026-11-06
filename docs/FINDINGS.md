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

## 8. B depth estimator, round 2: what fixed it (2026-11-05)

Diagnosis: profiled a dark-gap patch next to the needles over all 163 frames.
- The gap is darkest (luma 0.083) in frames 0–18, when the needles are in
  focus and their light stays confined.
- It brightens steadily to 0.264 by frame 162 as defocused needle light spills in.
- Its RAW focus measure rose 5.9× over the same range (brightness-dependent
  shot noise plus glow gradients).

So in a featureless region the raw argmax picks the frame with the MOST spill.
That is the B halo mechanism.

Fixes, now the default (`depth_method="cvf"`):
1. **Contrast-normalised focus measure**: Laplacian energy / (local mean² + 0.02²).
2. **Cost-volume filtering**: each frame's normalised FM is aggregated with a
   guided filter (guide = pass-1 C luma, radius 24 full-res px) before the
   running argmax. Evidence spreads within an object; featureless areas
   inherit a coherent depth. Order-independent, so it fits the multi-GPU fold.
3. **Soft-label compositing**: integer labels; at boundaries the two SIDES'
   frames are blended (blurred label indicators, σ 3 full-res px). The old
   Gaussian blur of the depth passed through intermediate frames with nothing
   in focus, which drew dark seams.

Result:
- Real stack: dark gaps stay dark, background patchworks and outlines are
  gone, and B looks close to Helicon B at half res.
- **Cn (noise-aware pyramid) is now close to the user's hand-made B+C
  composite** (`~/Desktop/stacker_vs_helicon.jpg`, `stacker_Cn_halfres.jpg`).
- Synthetic B: 36.2 / halo 31.8 at radius 6 (was 35.9 / 31.5).

Residual: radius trade-off. Small radii (6–8) are best on synthetic and sharpest
at depth edges, but leave dark blotches inside bright low-texture crystal on real
data (normalisation favours darker frames where texture is weak). 24 is the real-
data default. One faint background boundary remains at half res; the fan centre
is a touch hazier than Helicon B at 1:1.

## 9. B depth estimator, round 3: regularised depth (2026-11-06), now the default

Travis's verdict on round 2: "B is terrible". The cost-volume filter's single
radius smeared depth across fine 3D structure (soft matrix, dull crystals) and
left blotches and terraces in the defocused background.

**Reverse-engineering Helicon's depth.** For every region of Helicon's B
output, find the best-matching source frame. The match residual is 0.0024 RMS,
about the noise level, so Helicon composes almost purely from single frames.
- **Background: Helicon picks the DARKEST frame** (70 % within 5 frames vs 11 %
  for the focus argmax). Defocus spill only adds light, so the darkest frame is
  the least contaminated.
- Background focus-curve contrast is 0.39–0.51; texture is 0.66–0.69.

**Pipeline** (`depth_method="sgm"`):
1. Cached volume of contrast-normalised focus + local brightness per frame, at
   1/2 of the output grid.
2. Data cost = w·(1 − FM/FMmax) + (1 − w)·darkness, with w ramped over focus-curve
   contrast 0.45 → 0.60.
3. **Semi-global matching**: 8 directions, P1 = 0.03, P2 = 0.4 reduced across
   image edges.
4. Edge-guided upsampling, then a **refinement pass**: fine-scale focus argmax
   within ±6 frames of the SGM depth, where confident.
5. **Dual soft-label composition**: 3 px and 40 px label blurs. The wide blend
   applies only where focus confidence AND image texture are both low (texture
   = brightness-normalised |Laplacian| of C: background 0.015–0.021, matrix
   0.046–0.113; smooth crystal faces keep the narrow blend through their focus
   confidence).

Results:
- Background agreement with Helicon depth: 8 % → 69 %.
- Sharpness (mean |Laplacian|) vs Helicon B: crystal fan 0.96, crystal edge
  1.05, matrix 1.11. Helicon C reads 0.97–1.00 on the same measure.
- Synthetic B: 36.3 / halo 32.0 / plateaus 44.4 (best yet).
- Depth experiments run from the cache in ~6 s (`depthlab.depth_only`); a full
  half-res fusion takes ~2.5 min on the Mac.

Residual: a faint lighter, ghosted zone in the background just right of the
specimen; crystal-fan needles are slightly less crisp than Helicon B at 1:1.

## 10. "Glare spots" in Cn, and Helicon's post-processing (2026-11-06)

- **Helicon's B and C are darker than EVERY source frame on ~50-55 % of the
  image** (local means vs the per-pixel min over all 163 frames). No composite
  can do that, so Helicon applies a tone curve: fitted Helicon ≈ 1.06–1.10 ×
  ours − 0.028 (lower black point, more contrast). Ours stays within the
  frames' range. A large part of "less glare" is this global curve: our shadows
  sit ~0.02–0.03 higher (veiling haze).
- Sparkle count (small bright peaks, tone-matched): Helicon B = Helicon C =
  878/Mpx; ours 1150–1280, **including our B**, which is built from single
  frames. So the sparkles are in the source frames, and Helicon's two methods
  share a common post-process. It is not denoising: background fine-noise std
  is the same (Helicon 0.0021, ours Cn 0.0022).
- Real Cn-specific glare: a milky glow by the clear crystal. Its source is
  max-energy selection at COARSE pyramid levels, which picks large defocused
  bright blobs.
- Tried: Cg (coarse levels from B) and Cd (fine levels locked to ±6 frames of
  B's depth, against specular glints from other depths). Neither reduced the
  sparkle count. Kept as variants for visual comparison.
- **B ↔ C slider** (`fusion.blend_bc`; CLI `fuse --save-pyramids` then `blend
  DIR t…`): per-level blend of the B and noise-aware-C pyramids, crossing
  over fine-first. 0 = pure B, 0.5 = B tones/coarse + C fine detail, 1 = Cn.
  It re-renders from stored pyramids without re-fusing, so in the GUI it is a
  live slider on the GPU.

## 11. B edge glare and a scale-dependence bug (2026-11-06)

Travis: "our B has too much glare around edges."

- Helicon's recovered depth is fragmented in the dark surroundings of bright
  objects (many small regions, mostly darker frames). Ours was smooth, so those
  surroundings sat at their own depth while the bright neighbour was defocused
  and spilled light on them: the classic depth-map halo.
- **Full-res test exposed a design bug.** The SGM pipeline measured focus and
  texture at the OUTPUT scale, and its thresholds had been tuned at half res.
  At full res nearly everything fell below them: washed-out, glowing patches
  over grains and crystals.

  Fix: the focus/brightness volume and the texture map are measured at a fixed
  1/4-sensor resolution whatever the output scale (`depth_full_fraction`). The
  fine refinement pass adds output-resolution detail. Re-measured thresholds
  landed where they were (contrast 0.45 → 0.65, texture 0.025 → 0.045).
- Added a spill-zone prior: pixels much darker than something bright within
  24 full-res px prefer darker frames even when confident. Small effect alone.
  A remaining bright wisp at one crystal edge is sharp, probably a real thin
  crystal edge that Helicon leaves defocused.
- Half-res result vs Helicon B (tone-matched): edge-band area markedly brighter
  4.5 % → 3.7 %, elsewhere 3.6 → 2.9 %; sharpness up (fan 1.25, edge 1.25,
  matrix 1.09); defocused background now looks like Helicon's. At full res the
  fixed SGM is close to Helicon B on grains and crystals
  (`~/Desktop/stacker_B_edges_fullres.jpg`).
- Residual: a few small dark specks in the far background.

## 12. B: out-of-range regions, blend width, doubled edges (2026-11-06)

Travis: "still a lot of glare and doubled edges in the crystal fan and edges;
background edges look way worse compared to Helicon."

- **The "background edges" were regions out of focus in EVERY frame**, beyond
  the stack's depth range: 11 % of this image, mostly peaking at the first frames.
  - Their focus curves are monotonic ramps. A high peak-vs-median "contrast"
    fooled the confidence, and the refinement turned noise into a mottled, lacy
    patchwork.
  - Fix: a curve peaking within 3 frames of either end gets zero focus
    confidence plus a pull toward that end frame (the least-defocused view;
    Helicon shows those regions as soft but structured grains). The darkness
    prior is limited to featureless pixels: the two priors fought and made
    "bubbles".
- **Blend width now follows fine detail actually present** in the narrow
  composite (output-res |Laplacian| vs its own noise floor). Blurry regions use
  a CONTINUOUS smoothed depth, interpolating the two bracketing frames, so blur
  changes gradually and label seams cannot draw outlines. Quarter-res texture
  had counted defocused blobs as texture.
- **Doubled needle edges** persist even with hard labels, so they are not a
  blending artefact. A defocused thin bright needle images as two parallel
  lines; the chosen frame is slightly off that needle's focus.
  - Trusting strong fine-scale peaks at any depth did not fix it.
  - Locking scale did not fix it (near the image centre registration is not the cause).
  - Still open: a smaller white glint blob remains, and one needle edge is still doubled.
- Metrics vs Helicon B (tone-matched, half res): edge-band glare 3.7 % → 3.3 %;
  sharpness 1.05–1.16. The out-of-range background now looks like Helicon's.

## 13. B: dark background behind bright crystals (2026-11-06)

Travis: "Helicon's B handles dark background behind bright crystal much better,
ours still has ghostly glow and doubled edges."

Mechanism: dark background pixels next to a bright crystal were composed from
frames focused on the background, where the crystal is defocused. Its blur adds
glow, and the rim of its blur disc reads as a "sharp" ghost edge, so the focus
measure is confidently wrong there. Helicon uses frames where the crystal is
in focus, which leaves the dark featureless background dark.

Fix (the spill zone): dark pixels near an EXTENDED bright object get their focus
confidence discounted by h, a darkness preference (×2), and no fine refinement.
Five iterations to get the gate right:
1. 24 px reach: too short (defocus blur here is ~60 px).
2. 80 px with wide smoothing of h: cleared the background, but spread onto the
   crystal's own edge (soft, bubbled edge).
3. Light smoothing + erosion: crystal edge crisp. But with no gate, dark
   textured matrix near sparkles turned waxy (matrix sharpness 1.05 → 0.71).
4. Gates on texture and brightness swing: matrix fine, but the gap's glow came
   back. Its own texture (0.07) overlaps the matrix's, and the composite's glow
   makes those pixels look brighter, so they fall out of the zone.
5. **Final: the bright object is found by morphological opening** (removes
   bright features under ~16 full-res px: sparkles and grain highlights), so
   the matrix never sees "bright neighbours". The rule then runs ungated, with
   only a high texture veto (0.20).

Result: the dark gap and background beside the crystal stay clean and dark, the
crystal edge is crisp, and the matrix is intact. Edge-band glare vs Helicon B
is 3.1 % (best yet); sharpness 1.07–1.24.

## 14. B: doubled plate edges ("ghost lines") (2026-11-06)

The specimen is tyuyamunite: thin plates whose edges, seen edge-on, image as bright lines.

- Scored against Helicon's actual output (tone-matched), the current B is closer
  than any Helicon-style "simple" B (per-pixel argmax over radius 9 with
  Laplacian / normalised / gradient / variance measures: error 0.0255–0.038 vs
  0.0226, edge glare 4–7 %). So the multi-stage pipeline stays.
- Frame strip at a plate cluster: edges are crisp at frames 107–113 and double
  from ~116 (the rim of a thin bright edge's defocus disc). Our B took ~118.
  - Every focus measure, ours included, peaks at 110–112 when AVERAGED over
    the plate.
  - Per pixel, the pixels BESIDE an edge peak in the frame where the doubled
    ghost line passes over them, so the composite painted ghost lines next to
    real edges.
- The recovered "Helicon depth" disagrees with all our measures at fine scale.
  It matches Helicon's output against OUR aligned frames, so any alignment
  difference biases it: use it for regions, not for fine structure.
- Fix: edge-aware (guided-filter) aggregation of every frame's focus slice over
  32 full-res px before SGM (`vol_agg_px`). That spans the ghost offset, so
  pixels beside an edge inherit its in-focus evidence. Fine refinement widened
  to ±12 frames with a 9 px window (Helicon's "Radius 9" scale).
- Result: edge-band glare vs Helicon B 3.1 % → **2.3 %**; edge-band error 0.0204
  → 0.0199. The fan's glint blob shrinks to a small highlight like Helicon's;
  plate outlines are closer to single. Matrix and backgrounds unchanged.

## Open items

- **Residual flat-target roll bias**: a smooth hump of ±17 mdeg RMS per frame on far-defocused frames, antisymmetric about focus (the fitted rate is fine: 52 vs 50 mdeg/mm). Also ε reads −2.4 % (≈ 6σ). Cause unknown.
- Validate everything on **real** stacks:
  1. a flat-target stack (the diagnostic);
  2. a specimen stack (how big is the relief bias in reality?);
  3. the same specimen with scale locked vs free.
- Test the reversal protocol.
- B: ghosted lighter zone right of the specimen; fan needles a touch less crisp
  than Helicon B at 1:1 (see 9).
- Optional output tone stage (black point / contrast) to match Helicon's look,
  and optional highlight/sparkle suppression (see 10).
