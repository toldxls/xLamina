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

## 15. M0 on the PC: measure before optimising (2026-11-06)

Machine: Ryzen 9 7950X3D, 192 GB DDR5, ASUS ProArt X670E-Creator, Windows 11,
driver 621.01, CUDA 13.5, MSVC 19.44. `tools/m0/m0_bench.cu`, built with
`tools/build.cmd`; raw numbers in `tools/m0/results/`.

- **Cards.** RTX 3080 Ti in PCIEX16_1 at **Gen4 x8** (the BIOS has PCIEX16_2 set
  to x8 mode, which halves slot 1). RTX **3070** (plain, 8 GB; not the 3070 Ti in
  DESIGN.md) in PCIEX16_2 on a riser cable that only links at **Gen3 x8**: the
  card was invisible on Auto and forced Gen4 and appeared at Gen3.
- **Python reproduction.** `score_b.py` gave the Mac's numbers exactly (RMSE
  0.0228 / 0.0199, glare 2.3 %, depth 46 %); alignment transforms agree to
  3e-6 px. 13 tests pass, 1 xfail. Analysis 6 min, half-res fusion ~3.5 min on
  the CPU (vs 3 / 2.5 min on the M3 Pro).

Per frame (6960 × 4640 × 16-bit RGB = 193.8 MB), median of 10:

| | 3080 Ti (Gen4 x8) | 3070 (Gen3 x8) |
|---|---|---|
| H2D pinned / pageable | 13.3 / 13.0 GB/s (14.5 ms) | 6.7 / 6.6 GB/s (29 ms) |
| D2H pinned | 13.2 GB/s | 6.6 GB/s |
| uint16 → luma | 0.40 ms | 0.80 ms |
| similarity warp (texture) | 0.32 ms | 0.64 ms |
| focus measure (blur 0.8 → Laplacian² → blur 3) | 2.24 ms | 4.08 ms |
| pyrDown level | 0.20 ms | 0.40 ms |
| those four together | 3.2 ms (317 fps) | 5.9 ms (169 fps) |

Host: memcpy 23–26 GB/s whatever the thread count (dual-channel DDR5, one
crossing). RAM disk (Q:) **16 GB/s** unbuffered with 8 or 16 reader threads
when every read lands in fresh memory (a first version reused a 16 MB buffer
per thread, which stayed in the 7950X3D's L3 and read a flattering 20 GB/s);
the Windows file cache reads at the same rate. The NVMe source folder reads
at 6.5 GB/s. Both GPUs uploading at once: no interference, each keeps its own
rate (13.3 + 6.7 GB/s).

What it means:
1. **Both cards are PCIe-bound, by 4–5×.** The 3080 Ti computes a frame's
   kernels in 3 ms and uploads it in 14.5 ms. A second GPU helps because it adds
   a link, not because it adds compute. The 3080 Ti at x16 would double its
   upload rate; that needs the 3070 out of PCIEX16_2.
2. **Stack upload for 163 frames: ~1.6 s** with both links busy; reading from the
   RAM disk ~2 s; kernels ~0.4 s. So one pass over the stack is ~2–2.5 s if
   reads, uploads and kernels overlap (confirmed in §16: 2.4 s). Helicon's B at
   20 s and C at 10 s are beatable with room to spare; DESIGN's 3–5 s target
   for a 200-frame two-pass job is tight but plausible.
3. **Host memory is the next ceiling.** memcpy saturates at ~25 GB/s, so every
   extra copy of a frame costs ~8 ms, as much as a whole frame's kernels on the
   3080 Ti. The reader must `ReadFile` **directly into the pinned ring**
   (unbuffered, page-aligned), not read then copy; and the RAM disk should be
   read with ≥16 threads.
4. Uncompressed TIFF decode is a memcpy (DESIGN §3), so it is covered by the
   memcpy figure; no separate decode benchmark was needed.
5. Not measured yet: power-limited runs (`nvidia-smi -pl` needs admin) and
   per-card scaling with 2–3 streams; M2's scheduler gets those from the real
   fold kernels.

## 16. Port step 1: TIFF → pinned ring → GPU streaming (2026-11-06)

`core/` (stackcore: `tiff`, `pinned_pool`, `streamer`, `convert`) and
`cli/stackcli.cpp` (`probe`, `dump`, `stream`). Checked with
`tools/validate/check_read.py`.

- **The files.** Camera Raw's TIFF export is one uncompressed little-endian
  strip (193,766,400 bytes at offset ~20–24 K, after the XMP/EXIF/ICC blocks);
  Helicon's outputs are 4404 one-row strips with no gaps. Both read as a
  single `ReadFile`. The reader parses the first IFD itself (classic/BigTIFF,
  either byte order, uncompressed chunky 8/16-bit); compressed TIFF and JPEG
  are refused for now and will go through libtiff / libjpeg-turbo.
- **No copy on the host.** The read is `FILE_FLAG_NO_BUFFERING` straight into
  the pinned buffer, starting at the 4 KB boundary before the pixel data; the
  upload starts at the resulting offset inside the buffer. 16 readers, one
  shared frame queue, each GPU a worker thread with 3 streams and 3 device
  buffers; a frame's pinned buffer is released when its upload event fires.
- **Exactness.** `stackcli dump` vs `frames.read_rgb`: every pixel identical on
  frames 0, 81, 162 and on Helicon's B; whole-frame checksums agree to 1e-14.
- **Throughput, 163 frames (31.6 GB), unbuffered, into the convert kernel:**

  | GPUs | time | rate | split |
  |---|---|---|---|
  | 3080 Ti alone | 2.52 s | 64.8 fps, 12.6 GB/s | – |
  | 3070 alone | 4.89 s | 33.4 fps, 6.5 GB/s | – |
  | both | 2.41–2.57 s | 66–68 fps, **13 GB/s** | 83–90 / 73–80 |

  4, 8, 16, 24 or 32 readers and a larger pool make no difference: the RAM
  disk delivers ~13 GB/s while the GPUs are also DMA-reading host RAM (16 GB/s
  on its own, §15). That is the machine's ingest ceiling: **~2.4 s per pass**
  over this stack whatever the GPU count, and the split follows who is free,
  not the link ratio. The second GPU buys per-frame compute time (at 67 fps a
  frame has ~15 ms on the 3080 Ti or ~30 ms on the 3070), not ingest.
- The NVMe source folder streams at 6.2 GB/s (5.1 s): the RAM disk is worth
  2× here, which is less than the 3× its raw read rate suggests.

## 17. Port step 2: the similarity warp (2026-11-06)

`core/src/warp.cu` (`affine_for`, `box_down2_rgb`, `warp_bicubic_rgb`,
`load_aligned`), `stackcli warp`, `tools/validate/check_warp.py` against
`fusion.load_aligned` with the stack's own transforms
(`tests/data/report_2026-11-05.json`, native-crop grid), frames 0 / 81 / 162
at scale 1 and 0.5:

| | exact positions (product) vs cv2 | OpenCV-4 replica (1/32 px) vs cv2 |
|---|---|---|
| frame 81 (reference, identity) | identical, 100 % of pixels | identical |
| frames 0, 162, scale 1 | rms 1.2e-6, max 7e-5 | rms 7e-5, max 2.7e-3 |
| frames 0, 162, scale 0.5 | rms 5e-7, max 5e-5 | rms 6e-5, max 5e-3 |

- **OpenCV 5.0 (this PC) no longer snaps float warps to 1/32 px.** The
  prototype's `warpAffine` was reproduced to 1e-6 rms by sampling at exact
  positions with OpenCV's cubic kernel (a = −0.75), BORDER_REFLECT and the same
  float summation order, and *not* by replicating the 4.x fixed-point path.
  The §1 finding (1/32 px snapping biased pair scales ~20 %) was made with
  OpenCV 4 on the Mac; the alignment code's B-spline refinement makes the
  pipeline immune to it either way, which is why the alignment report
  reproduced to 3e-6 px across the two OpenCV versions (§15).
- The residual 1e-6 rms is OpenCV 5's own coefficient arithmetic (float32
  sample positions were tried: same gap). 7e-5 max = 4 LSB of 16-bit at a few
  high-gradient pixels, 0.07 LSB rms: far below anything the fusion scores
  can see (edge-band RMSE vs Helicon is 0.02).
- INTER_AREA ×0.5 as a 2×2 box in OpenCV's summation order matches to 2e-7
  (float association at the last bit).
- Cost on the 3080 Ti: **1.8 ms per full-res frame** (software bicubic, 16
  taps per channel, double positions) and 1.1 ms at half res including the
  box-down. The design allowed texture bilinear here for speed; software
  bicubic turned out cheap enough to keep the prototype's interpolation
  exactly, so it stays.

## 18. Port step 3: the pass-1 fold and the multi-GPU merge (2026-11-06)

`core/src/fold.cu` (focus measure, Laplacian pyramid, salience, the A / B0 /
C state updates, noise reference, host merge), `stackcli fold`,
`tools/validate/check_fold.py` (+ `make_list.py`). Per GPU: upload, convert,
warp and the per-frame measures run on 1–3 slot streams; one fold stream per
GPU applies the state updates in arrival order (events hand each slot's
scratch to the fold stream and back). The host merges the per-GPU states
with `_take`.

Against `fusion.fold` on frames 0 / 54 / 81 / 108 / 162 at half res:

| field | agreement |
|---|---|
| A (weighted average) | max 5e-5, rms 1e-6 (the warp's own tolerance) |
| focus measure (b_best, a_den, b_fmsum) | ≤ 5e-4 relative; rms 1e-8 |
| B0 index | 99.996 % of pixels; the rest are near-ties (candidates within 1e-3) |
| B0 colour, where the index agrees | ≤ 5e-5 |
| C index, per level | 99.99–100 % |
| C coefficient, where the index agrees | ≤ 3e-5 |
| C salience | within 2 √sal · 1e-4, as a coefficient error of 1e-4 implies |
| c_base, noise MAD, noise fm | 1e-6, 1e-5, 1e-6 |

**Invariant** (whole stack, GPU only): GPU 0 alone, GPU 1 alone, and both
GPUs with 3 streams produce **byte-identical B and C state**; A sums agree
to 1e-13 (double, different association).

Three things had to be learnt to get there:
1. **Ties must follow `_take` inside the kernel.** Frames reach a GPU in
   whatever order the readers finish, so a strict `>` let the first arrival
   win exact ties (common in salience: zeros, coarse levels) and C differed
   from run to run while B did not. `(v > best) || (v == best && k < idx)`
   makes the arrival order irrelevant.
2. **`cv::pyrUp` is not reflect-101 at the right and bottom.** It reflects
   101 at the top/left but uses `7 s[n-1] + s[n-2]` (even output) and
   `8 s[n-1]` (odd output) at the far edges. With reflect-101 everywhere, the
   last column and the last two rows of every Laplacian level were wrong.
3. **Salience tolerances are absolute, not relative**: a 3e-5 coefficient
   error moves a 1e-6 salience by 6 %.

The focus measure differs from OpenCV's by up to 5e-4 relative (two Gaussian
blurs and a Laplacian in float; summation order and FMA are not replicated).
It moves 0.004 % of B0's argmax choices, all near-ties, and is invisible to
the composites.

**Throughput, 163 frames, pass 1 complete:**

| grid | GPUs | time | frames/s |
|---|---|---|---|
| half res (3420 × 2214) | 3080 Ti | 2.59 s | 63 |
| half res | both | 2.45 s | 67 |
| **full res (6840 × 4429)** | **3080 Ti** | **2.8 s** | 58 |
| full res | both | 2.5 s | 65 |
| full res | 3070 (1 slot) | 8.1 s | 20 |

Pass 1 at full resolution runs at the ingest rate on the 3080 Ti (the
prototype takes ~3.5 min for the half-res fold on this CPU). Memory at full
res: 2.6 GB of state per GPU plus 2.2 GB per slot (raw bytes, float frame,
warped frame, measure scratch), so the 3070 holds one slot and cannot
overlap upload with compute; it contributes 47 of 163 frames beside the
3080 Ti. The slot count is now chosen per GPU from free memory.

## 19. Port step 4: the depth-map input volume (2026-11-06)

`core/src/volume.cu` (+ `imgops.cu`, the shared OpenCV-faithful primitives
the fold now uses too), `stackcli fold --volume`,
`tools/validate/check_volume.py`. In the same streaming pass as the fold,
every warped frame is resized (INTER_AREA) to the fixed depth resolution
(1710 × 1107 here, whatever the output scale), measured
(`focus_measure_rgb(normalized=True)`, blurred luma) and kept on its GPU as
float16 slices. After the merge, the first GPU collapses the merged C
pyramid into the guide (luma, clipped), resizes it to the depth grid, and
each GPU aggregates its own slices with the guided filter (r = 8, ε = 1e-4),
clamps and square-roots them; the host gathers plane-major `V` and `Lv`.

Checked on frames 0 / 54 / 81 / 108 / 162 at half res, each operation on
the same inputs the prototype would see:

| operation | agreement |
|---|---|
| fm, lum slices (float16) | identical at 94–99 % of pixels, max 1 ulp (1e-3) |
| pyramid collapse (same state) | max 2e-7 |
| INTER_AREA resize (same guide) | max 6e-8 |
| guided-filter aggregation (same slice + guide) | max 4e-5, rms 4e-8 |
| V end to end vs all-Python | rms 1e-5; 0.0001 % of pixels beyond 1e-2 (C near-ties, §18) |

- `cv::resize` INTER_AREA has two paths: integer factors sum the cells
  (half res: exactly 2 × 2); anything else, e.g. 4429 → 1107 at full res
  (factor 4.0009), uses weighted overlaps (`computeResizeAreaTab`). Both are
  reproduced with OpenCV's float summation order.
- The per-frame slice costs ~0.5 ms on the 3080 Ti and hides inside the
  ingest budget: pass 1 with the volume on is **2.5 s at half res and 2.6 s
  at full res** on both GPUs (3.2 s on the 3080 Ti alone), unchanged from
  the fold without it. Guide + aggregation + gathering the 2.4 GB of volumes
  to the host adds 0.65–0.9 s, almost all transfers, done serially per slice
  (an obvious later optimisation; SGM will want `V` on one GPU anyway).

## 20. Port step 5: cost volume, SGM, sub-frame depth (2026-11-06)

`core/src/depth.cu` (`depth_from_volume`), `stackcli depth`,
`tools/validate/check_depth.py`. From the aggregated volume on one GPU:
per-pixel curve statistics (max, median over K, argmax; Lv min / max),
the spill zone h (ellipse opening r 4, dilation r 20, texture veto, blur,
ellipse erosion, all OpenCV-shaped), confidence w with the end-frame mask,
the cost C in place of V (numpy's float64 promotions reproduced where they
happen), P2 from the guide's Sobel gradient, SGM, parabolic sub-frame
argmin, bilinear upsample, guided filter on the output grid (r = 2 ow/dw),
rint, 3 × 3 median; conf = w upsampled.

- **SGM on the GPU:** one thread block per scanline, one thread per label,
  the previous pixel's costs in shared memory with ±∞ guards, a block-wide
  min per step; diagonals enumerate their entry row and column. Directions
  run in the prototype's order so S accumulates identically.
- Checked on a 300 × 200 × 163 crop of the real volume (both sides get the
  GPU's V, Lv and guides; the numpy reference takes 0.4 s on the crop and
  minutes on the whole volume): **spill h and conf to 7e-7; depth_s
  identical at 99.999 % of pixels, the rest off by one frame.**
- Whole volume (1710 × 1107 × 163) on the 3080 Ti: cost 356 ms (the
  per-pixel median sort dominates; a selection would halve it), **SGM
  87 ms**, post-processing 20 ms at half res / 70 ms at full res; 0.5 s in
  all, 0.7 s on the 3070. 19.4 % of pixels peak at an end frame.
- Memory: V (1.2 GB) is overwritten by C, S adds 1.2 GB, Lv 0.6 GB: ~3 GB
  on the depth GPU, which at full res means freeing the pass-1 slots first.

## 21. Port step 6: refinement, composition, finalize — `stackcli stack` (2026-11-06)

`core/src/compose.cu`, `stackcli stack LIST --rect … --out DIR [--scale]`:
the whole fusion as one command, writing A, B, B0, C, CB, Cn, CBn, Cd, Cg,
Cg2 and H as 16-bit TIFFs (own writer, one strip) plus the depth maps.

Pipeline on the PC, in order: pass 1 (fold + volume slices, all GPUs) →
depth on the primary GPU → **pass 2** (refinement: fine focus argmax within
±12 frames of the SGM depth, all GPUs, `_take` merge) → refined depth
(`median3`, conf and spill gates) → smoothed depth (two `median5`, guided
filter r = 40 scale) → **pass 3** (dual composition: blurred narrow labels
and the continuous smoothed depth, double accumulators per GPU, merged by
summation) → mix by fine detail (20th-percentile noise floor) and focus
confidence → noise σ per level from two adjacent middle frames → finalize
(pyramid variants on the primary GPU) → TIFFs.

| | half res (3420 × 2214) | full res (6840 × 4429) |
|---|---|---|
| pass 1 + volume | 3.5 s | 3.3 s |
| depth | 0.8 s | 0.9 s |
| refine pass + refined/smoothed depth | 3.5 s | 4.2 s |
| compose pass + mix | 3.6 s | 4.8 s |
| noise + finalize | 0.5 s | 1.3 s |
| write 11 TIFFs | 0.5 s | 1.9 s |
| **total** | **14 s** | **18.7 s** |

Helicon on the same card: B ~20 s, C ~10 s, one output each. The prototype
needs ~3.5 min for the half-res run on this CPU.

- Three passes over the frames are inherent: the composition needs the
  final depth, which needs every frame's fine focus measure, and the stack
  (59 GB warped at full res) cannot stay on the GPUs. Each pass costs the
  ~2.5 s ingest (§16); what remains outside the passes (~5–6 s at full res)
  is host transfers, host merges and serial post-processing, all trimmable.
- The composition pass streams only frames whose label is used; on this
  stack the smoothed depth spans all 163.
- Memory at full res: the 3070 fits one slot per pass beside the state, so
  it takes a third of pass 1 and ~40 % of the later passes.
- **End-to-end against the prototype** (`tools/validate/check_stack.py`,
  half res, the Python run regenerated on this PC: 221 s for the fusion):

  | output | max | rms | within 1 LSB₁₆ | within 4 LSB |
  |---|---|---|---|---|
  | A | 1.5e-5 | 1.7e-6 | 100 % | 100 % |
  | B | 7e-3 | 5e-5 | 97.7 % | 98.8 % |
  | B0 | 0.31 | 3e-4 | 99.98 % | 99.99 % |
  | C | 5e-2 | 9e-5 | 99.8 % | 99.9 % |
  | Cn, Cg, Cg2 | 4e-2 | 6e-5 | 97.6 % | 98.9 % |
  | CB, CBn, H | 5e-2 | 1.3e-4 | 91 % | 95 % |
  | depth | identical 99.87 %, within 1 frame 99.97 %, within 3 frames 99.99 % | | | |

  The large maxima are single pixels where a near-tie (§18) or a one-frame
  depth difference chose another frame; the depth-guided variants inherit
  both. **Scored against Helicon B exactly like `score_b.py`, the GPU's B
  gives RMSE all 0.0228, edge-band 0.0199, edge glare 2.3 %, depth~Helicon
  46 %: the handoff's baseline numbers, identical to the prototype's.** The
  port reproduces the prototype.

## 22. Port step 7: alignment on the GPU (2026-11-06)

`core/src/align.cu`, `stackcli align DIR --out DIR`: pass 0 streams the
frames and keeps each one's luma at the align scale (INTER_AREA ×¼) on the
host; then every GPU pulls pairs from a shared counter, each with an LRU
cache of 24 frame pyramids (blur 0.8 + INTER_AREA halving to 4 levels,
Sobel-1 gradients, level-0 cubic B-spline coefficients). Per pair: phase
correlation at the top level (Hanning window, OpenCV's padding and 5 × 5
centroid; the DFT is two twiddle matmuls, since Windows has no static
cuFFT), Gauss-Newton per level with the OpenCV-style bicubic warp
(replicate border, exact positions), then 4 iterations at level 0 with the
exact B-spline warp (scipy `spline_filter` / `map_coordinates`, mirror).
The 6 × 6 normal equations come from a fixed-order block reduction
(deterministic); the host solves, iterates, and computes the covariance
and NCC from the last iteration's sums like the prototype. The global solve
(`solve_global`, IRLS, Huber 3), the reference choice, the rebase and
`common_rect` are ported on the host; the output is `report.json` in the
prototype's format plus `list.txt` for `stack`.

Against the Mac's report on the 163-frame stack: **max shift difference
3e-6 px, scale 2e-9, rotation 1e-4 mdeg, corner error 1e-5 px**; the same
crop (6840 × 4429, L29 R91 T88 B123), 637/637 pairs, 0 outliers, identical
NCC min / median, scale spread 2.5109 % both. That is the Mac-vs-PC-Python
level of agreement (§15), so the GPU pairs converge to the prototype's fixed
points.

- Time: luma pass 2.7 s, 637 pairs 6.3 s on both GPUs (10 ms per pair,
  393 / 244 split), solve + report < 0.1 s: **12 s**, against ~6 min for
  the prototype's multiprocess alignment on this CPU.
- One trap: the mirror initial condition of the B-spline prefilter divides
  z^(n-1) by z^i, which underflows to 0 / 0 on 1740-sample lines; Unser's
  truncation at the precision horizon (27 terms for the cubic pole) is what
  scipy does too, and fixed the NaNs.
- Not ported: the optical-train diagnostics (`diagnostics.py`: ε, tilt,
  FOE, jitter, lead-screw) and the plot. They are statistics on the
  per-frame (z, t) and can stay in Python for now (`report.json` is
  compatible) or be ported with M1.

## 23. `stackcli run DIR`: folder in, stack out (2026-11-06)

The end-user command chains the alignment (§22) and the fusion (§21) with
the transforms kept in memory: `stackcli run Q:\TIFF --fov-mm 0.98
--outputs B,C,Cn` writes `report.json` (the prototype's format, now with
the calibration block: µm/px, its range over depth, FOV, corner shift) and
the chosen 16-bit TIFFs into `DIR_stacker` or `--out`. `--lock-scale 1`,
`--ref`, `--scale 0.5`, `--gpus` pass through. 163 frames at full res:
**27.7 s** (alignment 10 s, fusion 17.6 s), the report identical to the
Mac's including the geometry numbers.

- A CUDA trap worth knowing: a kernel with a large per-thread local array
  (the B-spline prefilter's `double[4096]` line) makes the driver reserve
  local memory for every thread that could be resident (80 SMs × 2048 ×
  32 KB ≈ 5 GB on the 3080 Ti) and never return it; the fusion after the
  alignment then ran out of slots and the depth stage took 4.5 s instead of
  0.9 s. Lines now live in a global scratch buffer. (`k_curve_stats` keeps a
  2 KB local array: 335 MB reserved, tolerable.)

## 24. C's noise and the Smoothing control; colour (2026-11-06)

Travis: "C looks very good but we lack a smoothing option, Helicon comes out
much less noisy". Measured as the robust std of the high-pass residual (σ = 2
Gaussian removed) over the flattest 30 % of the image, per luminance band,
in 1/255 units:

| image | dark | mid | bright |
|---|---|---|---|
| C (raw argmax) | 1.08 | 1.52 | 2.08 |
| Helicon C, Smoothing 1 | 0.61 | 0.71 | 1.18 |
| B | 0.44 | 0.63 | 0.98 |
| Helicon B, R9 S1 | 0.47 | 0.57 | 1.00 |
| Cn | 0.42 | 0.63 | 1.05 |
| one source frame | 0.58 | 0.84 | 1.36 |
| C, smoothing 1 / 1.5 / 2 | 0.66 / 0.53 / 0.46 | 1.06 / 0.87 / 0.74 | 1.70 / 1.50 / 1.31 |

B and Cn are already at Helicon's level; raw C is twice as noisy as a single
frame because in flat areas the argmax picks whichever frame's noise is
largest. The fix is Cn's noise gate at a chosen strength: pyramid detail
below `smoothing × σ_level` is taken from B's coefficient
(`FinalParams::c_smoothing`, `--smoothing`, the GUI's "Smoothing (C)").
0 keeps the raw selection (the validated reference C), 2 matches Helicon's
Smoothing 1 (and keeps the plate edges intact at 1:1), 3 is Cn by
construction. The GUI defaults to 2, the CLI to 0.

Colour: the frames carry an Adobe RGB ICC profile; the outputs now embed it
and the GUI converts to sRGB for display (`QColorSpace`), which was the "far
off" colour. The remaining difference to Helicon is its tone curve (§10).

## 25. C's ghost outlines are defocus rims from far frames; the C radius (2026-11-06)

Travis: the plate edges in C (and H, CB, CBn, Cn, which share C's selection)
carry doubled rims and parallel "outlines" 10–40 px outside the true edge, and
"needles" inside the plate that are not real. Checked on the top edge of the
lower-left plate (output px 1400–2000 × 3050–3350) by re-doing C's per-level
argmax on the warped source frames and dumping the chosen frame per level:

- The edge itself is chosen from frames ≈ 85 at every level.
- The band 5–40 px outside the edge, in the dark background, is chosen from
  **frames 120–140 at levels 2–4**: 40–50 frames away. As the camera moves away
  the plate's defocus disc grows outward, so a background pixel at distance r
  from the edge always has its strongest mid-scale energy in whichever far
  frame puts the rim of that disc on it. Max-energy selection cannot tell a
  rim from an edge. Level 0 in the band is noise (random frames).
- Cross-level consistency (coarse levels forced to agree with the fine ones)
  does nothing: the fine levels there are noise or the same rim.
- Two rules remove the rim on this crop: (a) aggregating each frame's energy
  spatially before the argmax (Gaussian, σ 24–48 sensor px at level 0, halved
  per level, levels ≥ 1), so the edge's own frame wins the band beside it and
  shows it dark; (b) "ownership": a pixel whose best energy is < 0.3 × the
  strongest pixel within 64 px takes that pixel's frame. (a) fits the
  streaming fold (per-frame, order-independent merge unchanged); (b) would
  need a second pass because losing frames' coefficients are not kept.
- Implemented (a) as `FoldParams::c_radius` / `--c-radius R` / GUI
  "Radius (C)" (CLI default 0 = the validated reference C; GUI default 32).
  No measurable cost: 163 frames at full res still run in 27.5 s.
- Real stack, full res, radius 0 / 32 / 64 vs Helicon C: the doubled top edge
  and the glow above it are gone at 32 and 64, matching Helicon. The stacked
  parallel outlines at the plate's lower-left corner **remain at every
  radius**, as they do in Helicon's C: there the rims lie inside the bright
  plate, where every pixel's own strongest energy is a rim from some other
  frame, so no nearby pixel is strong enough to own them. Those need a
  different rule (frame-distance consistency to the nearest edge, or B's
  depth once B's spill handling is fixed).

### 25a. The C radius reaches B through the guide; B's own radius and priors (2026-11-06, later)

- **B changes with the C radius**: the guide for B's edge-aware aggregation,
  SGM edge penalty, guided upsampling and depth smoothing is the luma of the
  collapsed C (`run_pass1`). On the frozen 163-frame list, B at C radius 32
  vs 0 differs on 51 % of pixels by > 4 LSB₁₆ and on 8 % by > 1 % of range
  (radius 0 twice: bit-identical). Not visibly better or worse at the plate
  edge. Anyone scoring B against the baseline must run with `--c-radius 0`.
- **B's doubled plate edge is not fixed by the same trick.** `--b-radius R`
  (isotropic Gaussian over each aggregated focus slice, R sensor px) at 16 /
  32 / 64 and `--halo-prior` 4 / 8 (the darkness preference beside bright
  objects, default 2) each trim the haze above the edge a little; the second
  edge line survives in all of them, while Helicon B shows one line. Reason:
  B's focus measure is contrast-normalised (lap² / (mean² + 0.02²)), so the
  rim on the dark background scores as high as the bright edge itself and
  spreading the edge's energy cannot outvote it, unlike C's raw energy. The
  controls stay (CLI, GUI "Radius (B)", default 0) for tuning; the fix needs
  rim rejection in the volume (fine-scale confirmation of the coarse peak) or
  depth ownership by the nearest strong edge.

## 26. Resident frames: the stack pinned in host memory (2026-11-06, late)

Trigger: the 3070's riser was fixed and both cards now link at Gen4 x8 (0
PCIe replays under load), but the stack got no faster. Timed with the CLI on
the 163 frames at `Q:\`, full res, outputs B,C,Cn:

| | both GPUs | 3080 Ti alone |
|---|---|---|
| alignment pairs | 6.4 s | 9.4 s |
| pass 1 stream | 2.5 s (64 fps, 12.5 GB/s) | 3.2 s (51 fps) |
| pass 2 fine focus | 3.1 s | 4.5 s |
| pass 3 composition | 2.4 s | 2.5 s |
| fusion total | 16.3 s | 16.2 s |
| whole run | 26.2 s | 29.3 s |

The second card bought 3 s of alignment and nothing in the fusion: every
streaming pass re-read the 31.6 GB stack from the RAM disk, which the host
delivers at ~12.5 GB/s while the GPUs DMA from it (§16), so each pass was
~2.5 s whatever the links did. The two-GPU fusion also pays ~1 s for the
host merges and a slower compose, which ate the little the passes gained.

**The change** (`core/frame_cache`, `StreamerOptions::cache`): the first pass
that meets a frame reads it into a pinned buffer the `FrameCache` keeps;
every later pass uploads from that buffer and reads nothing. The ring is now
pinned lazily (an unused ring costs nothing). The budget is the smaller of
60 % of physical RAM and free RAM minus 8 GB, decided per job; frames past
it go through the ring as before, so a stack that does not fit still runs.
Entries are keyed by path and checked against file size and mtime. The GUI
owns one cache in its `Worker`, so a second render of the same folder uploads
from memory from pass 0 on; a job from the CLI gets one for its own duration
(`--cache 0` turns it off). The noise pair in finalize comes from it too.

Measured first (pinning 163 x 194 MB: 1.8 s at 17 GB/s, single- or
multi-threaded; freeing 1.5 s; H2D from pinned 13.3 GB/s on either card), then
on the pipeline, same frames, outputs bit-identical to the streamed run and to
the single-GPU run:

| | streamed (`--cache 0`) | resident |
|---|---|---|
| pass 0 (alignment luma) | 2.69 s read | 3.09 s read + pin |
| pass 1 | 2.75 s, 11.5 GB/s | 2.10 s, 15.0 GB/s |
| pass 2 fine focus | 3.32 s | 3.02 s |
| pass 3 composition | 2.68 s | 1.48 s, 21 GB/s |
| fusion total | 15.9 s | 13.1 s |
| whole run | 25.8 s | 23.5 s |

Pass 3 shows what the links can do (110 fps). Passes 1 and 2 are now bound by
their own kernels on the fold stream, not by ingest: pass 2's fine-focus
measure in particular (54 fps, the 3070 taking 58 frames). The next speed
work is there, and in the ~1.4 s of allocation and merge overhead in the
fusion, not in the I/O. A GUI re-render of the same folder also skips pass 0's
read (every pass "from memory" in the log).

## 27. Alignment pairs: float kernels, pairs in flight (2026-11-06, late)

After §26 the alignment's 637 pairs (6.5 s) were the largest block in the
run. Two changes, both checked against the double-precision report.

1. **Pairs in flight.** A pair's Gauss-Newton loop syncs the host every
   iteration (~21 iterations per pair: 6 at the top level, 3–4 at each lower
   one, 4 exact), so one pair at a time left the card mostly idle between
   kernels. Each GPU now runs `--pair-threads` pairs (default 3) on their own
   streams, sharing one pyramid cache per device (LRU past 16 + 8 T entries;
   an entry in use is never evicted; a thread that needs a pyramid another
   thread is building waits for it). The per-pair arithmetic is untouched,
   so the report is bit-identical for 1, 2, 3, 4 or 6 threads. Gain on its
   own: 6.46 → 5.85 s. The pairs were compute-bound, not latency-bound.
2. **Float kernels.** Nsight Systems on the 3080 Ti alone (637 pairs, 9.4 s):
   `k_gn_reduce` 46 % (0.55 ms per call at 1740×1160), `k_warp_bspline` 31 %
   (1.07 ms per call), `k_bspline_prefilter` 7 % (1.9 ms per pyramid axis),
   `k_final_reduce` 4.5 %. All ran in double on a GeForce card, where FP64 is
   1/64 rate. Now: the exact warp keeps the sample position in double and
   does the weights and 16 taps in float, with a mirror-free interior path
   (`align.h` always said FP32 weights; it was the texture units' 1/256 px
   *positions* that biased the scale, §1); the GN reduction accumulates
   float per thread over its ~32 pixels (grid capped at 256 blocks, fixed
   per-thread order) and double from the warp shuffle on; the prefilter's
   scratch is transposed so neighbouring threads hit neighbouring addresses,
   in 32-thread blocks (same arithmetic per line, bit-identical coefficients).

Result, both GPUs, 3 pairs in flight: **637 pairs in 2.77 s** (4.3 ms/pair),
alignment 9.7 → 6.0 s; on the 3080 Ti alone, one pair at a time, 9.4 → 4.6 s.
Against the double-precision report on the same stack: max |Δt| 1.4e-7 px,
|Δs| 5e-11 (0.2 nm at the corner), |Δθ| 3e-9°, the crop rect identical, NCC
identical to 6 decimals (the Mac–PC agreement in §15 was 3e-6 px). Two float
runs, and GPU 0 alone vs both, give identical reports. The fused outputs
move by ≤ 12 LSB₁₆ on < 0.06 % of pixels (B ≤ 3 LSB).

**Whole run now 19.2 s** (was 26.2 s at the start of the evening, 27.7 s in
§23): alignment 6.1 s, fusion 12.9 s. The remaining big blocks are the three
streaming passes' kernels (pass 2 fine focus 3.0 s, pass 1 2.1 s), compose's
host mixing (~1.5 s), the alignment read + pinning (3.1 s) and ~1.4 s of
allocation/merge overhead.

## 28. Pass 2: the Gaussian blurs (2026-11-06, late)

Nsight on the 3080 Ti alone, pass 2 (fine focus), per frame: upload 14.7 ms,
`k_blur_row` + `k_blur_col` 22.6 ms over the three blurs of `refine_measure`
(sigma 0.8, 9 and 27: the normaliser's kernel is 217 taps), warp 1.9 ms,
everything else < 1 ms each. The blur kernels read every tap from global
memory, so the card could not hide them behind the upload and pass 2 took
2.97 s with both GPUs.

`gaussian_blur` now runs shared-memory tiled kernels: the row filter loads
`1024 + 2R` pixels and the coefficients per block, the column filter a
`(32 + 2R) x 32` tile, and each thread produces four consecutive outputs from
sliding register windows, so a tap costs one shared load per two
multiply-adds instead of two per one (the first tiled version without the
register blocking was only 35 % faster: shared-bandwidth bound). The
per-output summation order is exactly the global-memory kernels' (taps in
increasing i; the column pair summed before its multiply), and the old
kernels remain as the fallback past 48 KB of shared memory (R > ~160) or
when R >= the image size. B, C and Cn are bit-identical to the previous
build, on one GPU and on two.

| both GPUs, 163 frames | before | after |
|---|---|---|
| pass 1 stream | 2.10 s | 1.77 s |
| pass 2 fine focus | 2.97 s | 1.41 s (116 fps) |
| pass 3 composition | 1.48 s | 1.41 s |
| fusion total | 12.9 s | 11.0 s |
| whole run | 19.2 s | **17.3 s** |

Passes 2 and 3 are now at the two links' upload rate; on the 3080 Ti alone
pass 2 is 2.45 s = 163 x 14.7 ms. What is left in the fusion: compose's
host-side mixing (~1.5 s of its 2.9 s), the pass-1 kernels (1.8 s; the fold's
pyramid and the volume slice), refine's host step (0.4 s), finalize (1.2 s),
and ~1.3 s of allocation and merges.

## 29. The host-side steps: merges on the GPU, finalize on demand (2026-11-06, late)

After §28 the fusion's streaming passes ran at the links' rate and the rest
was host work. Nsight (3080 Ti alone) showed what it was:

- **The fold state** (2.6 GB per GPU) was downloaded to pageable memory from
  every card, merged on the host, then uploaded again piecewise for the
  guide and for finalize. 727 MB pageable copies take 57 ms each way.
- **finalize** computed and downloaded all eleven outputs (11 x 363 MB
  pageable D2H plus the uploads of the state it needs) whatever `--outputs`
  asked for.
- **compose** downloaded the four double accumulators (1.94 GB per GPU),
  merged them on the host, uploaded them again for the mix.
- **refine** did the same with its 180 MB per GPU.
- **`k_fold_ab`** at 4 ms per frame was the biggest pass-1 kernel: 130 B/px
  of traffic, 64 of them for A's weighted double sums.

Changes, all with the same arithmetic in the same order:

1. Merges run on the first GPU. The other card's arrays cross through two
   pinned 64 MB buffers (`CrossCopier`: the D2H of one chunk overlaps the
   H2D of the previous) into a temporary device buffer, and `k_add_f64` /
   `k_merge_take` apply exactly `merge_into`'s rule. The merged `FoldState`
   stays on that GPU (`Pass1::dst`) for the guide and for finalize; the host
   copy is only made for the `fold` command (`host_state`).
2. `finalize` takes the device state and computes only the requested
   outputs plus what they depend on (H needs CBn; CB, CBn, Cd need the
   depth helpers; anything mixed with B needs B's pyramid). With B, C, Cn
   it builds two collapses instead of nine.
3. `compose_mix` and `refine_depth` read the merged device states directly.
4. The fold allocates and updates A's sums only when A is an output
   (`FoldParams::with_a`), which also shrinks the state from 2.6 to 1.6 GB.
5. The slot budget now reserves the depth-volume slices (K x dw x dh x 4 B),
   which the smaller state had exposed: the 3070 fitted two slots with
   0.9 GB left and the per-frame `cudaMallocAsync` was unchecked. It is
   checked now.

A bug found on the way: freeing the per-GPU states in a loop leaves the last
device selected; the next step then launched on the 3070 with the 3080 Ti's
stream and the process spun forever at 15 % GPU load. The primary is
re-selected after every such loop.

Checks: B, C, Cn bit-identical on two GPUs and on one; all eleven outputs
on the 3080 Ti bit-identical to the 0.5.4 build; the `fold` command's state
identical except `a_num` / `a_den`, which differ run to run in 0.5.4 as well
(the double sums follow the frames' arrival order; the A image is
unchanged). The merged B / C arrays are bit-identical across any split.

| both GPUs, 163 frames, B C Cn | §28 | now |
|---|---|---|
| pass 1 (setup + stream + merge + volume) | 2.4 s | 0.24 + 1.73 + 0.19 + 0.58 s |
| refine (pass 2 + host) | 1.84 s | 1.77 s |
| compose (pass 3 + host) | 2.94 s | 2.10 s |
| finalize | 1.2 s | 0.40 s |
| fusion total | 11.0 s | 8.2 s |
| whole run | 17.3 s | **14.6 s** |

What is left: the alignment (6 s: 3.1 s read + pin, 2.75 s pairs), pass 1's
kernels (1.7 s), the compose mix (0.7 s of host-visible GPU work), and the
depth stage (0.85 s).

## 30. The hardware sweep, and what the live view costs (2026-11-07)

The GUI's Tools > Benchmark (thorough mode) ran the 163-frame stack at full
resolution, outputs B C Cn, under every hardware setting varied one at a time
from the defaults (both GPUs, 2 streams/GPU, 16 readers, 3 pairs in flight,
frames resident). Total = alignment + fusion + report, no TIFFs written:

| run | total s | align | fusion | pass 1 | refine | compose |
|---|---|---|---|---|---|---|
| defaults, frames read from disk (cold) | 15.78 | 7.69 | 8.04 | 2.83 | 1.79 | 2.11 |
| defaults | 12.54 | 4.50 | 7.99 | 2.82 | 1.77 | 2.11 |
| GPU 0 only (3080 Ti) | 16.94 | 6.80 | 10.09 | 3.26 | 2.72 | 2.82 |
| GPU 1 only (3070) | 23.03 | 8.20 | 14.78 | 6.36 | 3.40 | 3.49 |
| 1 stream per GPU | 14.88 | 4.58 | 10.24 | 3.48 | 2.70 | 2.21 |
| 3 streams per GPU | 12.82 | 4.52 | 8.25 | 2.90 | 1.84 | 2.18 |
| 8 / 32 reader threads | 12.74 / 12.73 | | | | | |
| 2 / 4 pairs in flight | 12.92 / 12.51 | | | | | |
| frames streamed from disk every pass | 17.86 | 5.92 | 11.88 | 3.91 | 3.10 | 3.49 |

What it says: the two cards together stream pass 1 at 93 frames/s against
66 (3080 Ti alone) and 29 (3070 alone), the ideal sum; passes 2 and 3 run at
116 frames/s = 22.5 GB/s of uploads, the PCIe ceiling of two Gen4 x8 links.
Readers and pairs in flight make no difference once the frames are resident;
one stream per GPU loses the upload/compute overlap (+2.3 s); a third stream
gains nothing. So the defaults stand, and the only choices that matter on a
machine are *which GPUs* and *resident or not*, which is what the quick
benchmark (20 s: up to 40 frames at half resolution) now tests.

The GUI, though, took 13.0 s of fusion for the same stack with the Medium
live view, against 8.0 s quiet. The previews were the cause, in layers, each
measured on the GUI render (fusion seconds):

| the live view's plumbing | fusion |
|---|---|
| previews synchronise the slot stream inside the streamer callback (0.5.5; against the callback's own contract) | 13.0 |
| asynchronous: event per preview, delivered at the slot's next turn, blocking wait there | 10.8 |
| non-blocking: a frame whose predecessor is still in flight gets no view | 10.0 |
| 8-bit conversion moved to the GUI thread; pooled buffers; GUI-side 30 fps cap | 10.0 (no change) |
| preview kernel and copy on their own stream, only the kernel guarded | 10.0 (no change) |
| host hooks replaced by no-ops (GPU-side work kept) | 9.7 |
| Low (1/12) instead of Medium (1/6) | same as Medium |
| preview written by the kernel into mapped host memory (no copy engine) | 32.7 |
| views rate-limited: one per slot per 60 ms, one building map per 150 ms | **9.5** |

So the cost is on the GPU side and per copy, not per byte: any D2H copy
while the uploads saturate the link costs the uploads about 3 ms, whatever
its size, and nothing on the host side mattered. Rate-limiting is the right
tool (the eye does not need 90 frames a second); what remains (~1.3 s) is the
price of the live view, and "Off" is now a live-view setting for when the
time matters more than the show.

Two more savings on the way, both bit-identical (B, C, Cn compared byte for
byte against the 0.5.5 outputs after every step): the depth volume is
collected into page-locked host memory without the per-slice stream sync and
without the raw focus measure the fusion never reads (volume 0.59 -> 0.50 s;
the 1.8 GB is unpinned on a side thread, since unpinning took 0.2 s on the
critical path), and the output TIFFs are written on one thread each (0.68 ->
0.25 s). CLI fusion 8.65 -> 8.1 s. And the GUI's thumbnail loader, which read
the whole stack from disk while pass 0 was reading it too, is paused during
a render: alignment in the GUI 8.1 -> 6.5 s.

## 31. The depth volume at other resolutions (2026-11-07)

The GUI benchmark's synthetic sample now writes its truth beside the frames
(`truth/depth_frames_over_N-1.tif`, the depth per pixel as grey = z / (N - 1);
`truth/sharp.tif`, the all-in-focus scene; a README with the scene's formula),
and its defocus grows 1 px of sigma per frame of distance, about what a 20x /
NA 0.4 objective does at Travis's 0.468 um steps and 0.14 um/px. At the earlier
0.45 px per frame the depth was underdetermined at 1/4 resolution: 32 frames
differed by less than half a pixel of blur at the volume's scale.

The depth volume's parameters are now on the command line (`stackcli run ...
--radius-px --power --depth-fraction --depth-fm-px --vol-agg-px`, beside the
existing `--halo-prior --dark-prior --refine-r --refine-fm-px --soft-px`) and
in the GUI's Advanced panel, which gained Depth resolution (1/4, 1/2, full).

To run a 1/2-resolution volume on 163 frames at all, three memory changes went
in, all bit-identical for the default path (B, C, Cn compared byte for byte):
- pass 1's slot buffers (frame, warp and measure scratch, 2.2 GB each) are freed
  right after the streaming instead of at the end of pass 1, so the volume
  aggregation and the depth stage have the memory;
- the plane-major host volume is uploaded and transposed a few frames at a
  time through a 256 MB staging buffer instead of as a second full device copy
  (4.9 GB more at 1/2 resolution; 1.8 GB saved at 1/4);
- the SGM runs in overlapping horizontal bands (64 rows of overlap) when its
  path-sum volume does not fit beside the cost volume; a path's cost is capped
  by P2 at every step, so what it carries from rows farther away than that is
  nil. One band is the whole image, which is the code as it was.

The spill-zone radii now follow the depth resolution (they were fixed at 1/4).
A 1/2-resolution volume of 163 frames (4.9 GB of measure, 2.5 GB of luma, the
SGM in 8 bands) runs on the 3080 Ti alone in 13 s; the 3070 cannot hold the
slices beside its state. Full resolution would be 20 + 10 + 20 GB and needs
the depth stage streamed through the GPU in bands from host memory: not done.

On the synthetic stack the 1/2-resolution volume with a 3 px window, no
aggregation and no priors gave a depth map that was right up to a bright
plate's rim on both sides (transition +7..+15 px outside the rim, inside
band clean, 0.41 / 0.29 frames RMS on the matrix / plate), where the defaults
put the band 6-40 px inside the rim onto frames 9 away (the dark prior, which
also keeps the outside glow down) and spread the plate's depth 40-100 px into
the matrix in blotches (the 32 px aggregation and the 1/4-resolution grid).

On the real stack, scored per region against the reference B (plates, fan,
crystal edge, background edge, a flat matrix patch; each region registered
on its own, tone matched with a cubic, lumas compared after a 1 px blur):

| luma RMSE all / edges, depth RMSE frames | plates | fan | crystal edge | bg edge | flat matrix |
|---|---|---|---|---|---|
| default (1/4, priors, agg 32) | .0412 / .0766, 8.1 | .0299 / .0648, 12.2 | .0398 / .0879, 19.2 | .0657 / .1213, 3.4 | .0459 / .0804, 5.0 |
| 1/4, dark prior 0 | .0412 / .0766, 8.0 | .0298 / .0648, 13.0 | .0400 / .0880, 22.4 | .0657 / .1213, 3.4 | .0459 / .0805, 4.7 |
| 1/4, agg 8 | .0414 / .0772, 9.2 | .0301 / .0655, 11.3 | .0397 / .0878, 14.8 | .0657 / .1214, 4.2 | .0459 / .0805, 5.2 |
| 1/4, no priors, agg 0, window 3 | .0416 / .0775, 8.1 | .0319 / .0660, 10.8 | .0401 / .0884, 15.4 | .0657 / .1213, 5.4 | .0460 / .0806, 5.1 |
| 1/2, window 3 (priors, agg 32) | .0424 / .0763, **28.6** | .0301 / .0646, 22.1 | .0392 / .0868, 26.0 | .0661 / .1213, 11.3 | .0458 / .0802, 11.6 |
| 1/2, window 6 | .0425 / .0763, 29.0 | .0301 / .0647, 19.1 | .0393 / .0869, 23.5 | .0661 / .1211, 11.9 | .0459 / .0802, 9.8 |
| 1/2, window 3, no priors | .0409 / .0765, 10.0 | .0301 / .0644, 15.0 | .0398 / .0880, 22.5 | .0656 / .1211, 4.0 | .0460 / .0807, 5.2 |

- The synthetic result does not carry over as it stands. With the priors on,
  the 1/2-resolution volume sends whole plates to frames 11 away (depth bias
  +11 in the plates region): the priors' weights were tuned against the
  1/4-resolution focus term (sec. 11 again: thresholds tuned at one depth scale
  break at another). Without the priors the 1/2-resolution depth agrees with
  the reference about as well as the default does, no better, and loses the
  near-frame region at the crystal edge that the default finds.
- In the image every variant is within noise of the default (differences in
  the fourth decimal): the refinement and the soft composition absorb most of
  the depth differences.
- The 1/4-resolution "no priors, no aggregation" variant put a greenish patch
  into the fan region's dark background: the priors earn their keep there, as
  sec. 9 and sec. 12 said.

So the defaults stay. What this bought: the knobs on the CLI and in the GUI, a
1/2-resolution volume that fits, and a per-region scored comparison. The next
phase is the parameter search: many B runs over the depth resolution, window,
aggregation, the priors (rescaled with the resolution) and SGM's P1 / P2 (not
yet on the CLI), scored per region and judged at 1:1.

## 32. The depth volume at 1/2 resolution becomes the default (2026-11-07, late)

Travis rendered the stack with Advanced ticked and Depth resolution 1/2 (the
other Advanced values at their defaults: window 6, aggregation 32, dark prior
1, halo prior 2) and found the ghost outline of the plate in his screenshot
gone at 1:1, where every 1/4-resolution variant of sec. 31 (C radius 0 or 64,
B radius 16 or 48, halo prior 4, dark prior 2, refinement range 4, soft
labels 1.5, aggregation 64) still showed it. The 1:1 sheet of that crystal's
edge across all of them, with the reference B beside, is
`helicon_B_analysis/report_real/ghost_crystal.png` (outside the repo).

The per-region scores of sec. 31a had this variant within noise of the default
in the image and *worse* in depth agreement with the reference on the plates
(whole plates placed on frames 11 away), which the image did not show. Depth
agreement with another stacker's map is evidently not the criterion; the
eye at 1:1 is, and the doubled plate edge is the artefact Travis has flagged
since sec. 25. So 0.6.1 makes 1/2 the default depth resolution (engine and
GUI; `--depth-fraction 0.25` or the Advanced panel give the 0.5.x reference
B back). The prior weights are still those tuned at 1/4; rescaling them with
the resolution is part of the parameter search to come.

Costs: the depth stage 0.9 -> 2.8 s and the volume aggregation 0.6 -> 1.8 s,
about +3 s on a 163-frame render; 4.9 GB of focus measure and 2.5 GB of luma
in host and device memory. Memory: a card that cannot hold its share of the
slices beside its state (the 3070 at 1/2 resolution, when another process
holds part of it) now sits the job out instead of running out of memory
mid-pass, and the slice budget assumes 70 % of the frames per card when the
queue is shared. The archive's `score_b.py` baseline numbers refer to the 1/4
reference and no longer describe the default B.

## 33. The 1/2-resolution volume's blotches: the dark prior (2026-11-07, late)

With the 1/2-resolution depth (sec. 32) Travis found the plate edges fixed
but "blotchy garbage" elsewhere: a dark patch inside a needle fan, where the
matrix far behind the fan was rendered in place of the needles, and grainier
flat areas. On the fan region (output grid x 3480-4608, y 3152-3756), with the
reference B beside, in 1:1 sheets (`report_real/sgm_fan.png`, `fan2.png`):

| 1/2-resolution variant | fan blotch | plate ghost (sec. 32) |
|---|---|---|
| as sec. 32 (P1 0.03, P2 0.4, dark prior 1) | present | gone |
| SGM P1, P2 x2 (the grid is twice as fine, so the same penalty per pixel is half the smoothness per micron) | present | gone |
| P1, P2 x4 | present | gone |
| dark prior 0.5 | present | gone |
| halo prior 0 | present | gone |
| B radius 24 px | present | gone |
| **dark prior 0** | **gone** | gone |
| no priors, B radius 24 | gone | gone |

The dark prior ("prefer the darker frame in a featureless pixel", sec. 9)
is what fills the gaps between needles with the dark matrix frames once the
grid resolves those gaps; at 1/4 they were below the grid. It is the same
mechanism that put the band inside a bright rim onto far frames on the
synthetic stack (sec. 31). With it off at 1/2 resolution the four other
regions (dark background with a plate edge, crystal edge, plates, matrix
with a background edge; `report_real/dark0_*.png`) show no glow returning,
and the per-region scores against the reference are equal to the 1/4
default's or marginally better (plates .0409 / .0763 against .0412 / .0766).

Two scale fixes went in with it, both bit-identical at 1/4: SGM's P1 and P2
are multiplied by depth_fraction / 0.25 (`--sgm-scale 0` turns that off; P1,
P2 and the confidence ramp are also on the CLI now), and the post-SGM guided
filter's radius is 8 output px whatever the grid (it was 2 volume pixels,
which halved on the finer grid). So 0.6.2's defaults are: volume at 1/2,
dark prior 0, P1 0.06 / P2 0.8 in effect, guided radius 8. The 0.5.x
reference B is `--depth-fraction 0.25 --dark-prior 1` (still bit-identical
to 0.5.5).

Reproducibility of the banded SGM: a path's memory is cut at a band's edge,
so the band layout decides the result, and two runs with different free
memory gave different B's (16 against 12 bands). The path-sum volume now
gets a fixed 1.5 GB budget (4 bands at 1/2 resolution on this stack), and to
make that budget reliably available the luma volume (2.5 GB) is freed as
soon as the cost is built. Two consecutive default runs are byte-identical;
a card too short even for 1.5 GB falls back to more bands and says so in
the log.

## 34. The dark prior's reach, and defocused background on the 1/2 grid (2026-11-07, night)

Turning the dark prior off (sec. 33) traded the fan blotch for a worse
defocused background: Travis found 0.6.2's B "does poorly" there compared
with the 1/4-resolution B. Two problems, two changes, both in 0.6.3.

**The dark prior is back, with a reach.** It is what keeps featureless
background on consistent dark frames, and what it got wrong at 1/2 was not
the background but the gaps *between* needles and *inside* rims, which the
finer grid resolves as featureless pixels. So the prior now applies only to
featureless pixels with no confident (sharp) pixel within `dark_reach_px`
(24 sensor px; the confidence map dilated by that radius gates the prior
off). On the fan region the blotch is gone with the prior on; the plate
ghost stays gone; the other regions are unchanged
(`report_real/h3_*.png`). Alongside it, an out-of-range pixel (focus curve
peaking at an end frame) takes the end its neighbourhood votes for, the
vote smoothed over `end_vote_px` (48 sensor px), instead of its own noisy
peak's side. Two alternatives for the end decision were tried and dropped:
the luma trend of the curve (first third against last third) and the
darker end; both chose worse than the peak's side on the crystal edge.

**Defocused background broke into patches.** With the volume at 1/2, two
background regions (output grid x 5400-6300 y 1300-1900 and x 5600-6500
y 2700-3300) showed wispy contour lines that the 1/4 B did not have. Their
depth maps explain it: at 1/4 the whole area sits on frame ~2; at 1/2 it
splits into patches of frame 0 and frame 161 with hard borders, and the two
ends render the defocused foreground differently, so every border is a
seam. The reference B's map in the same area is a smooth gradient between
the two ends (median frame 11). None of the usual knobs fixed it: SGM P1/P2
x2 and x4, the confidence ramp (smears the plates), soft labels 6, the depth
window 9, the end vote at 96 and 200 px (slight help), focus trend. A
global isotropic blur of the focus slices (`--b-radius 128`) did cure it
(the area's depth median went to 2, like 1/4) but smeared the plates' detail
and altered the fan, so the smoothing has to apply only where the image is
flat.

The flat-region smoothing (`flat_blur_px` 128, `flat_reach_px` 24; GUI
"Flat smoothing"): a texture map of the guide at depth resolution (the
guide blurred by reach / 4 so its noise does not count, |Laplacian|, blurred
by the reach), ranked over the image: the flattest fifth is fully flat, the
midpoint to the 90th percentile and above is textured, in between a ramp.
A first version used the raw Laplacian with a fixed fraction of the 90th
percentile; the background noise alone exceeded it and the mask was empty
(mean 0.009 in the background), hence the pre-blur and the ranking. Where
the mask is set, (a) every aggregated focus slice is box-blurred over the
blur width (so the defocused background shares one curve over a wide area)
and (b, 0.6.3) the SGM depth was Gaussian-blurred over half the width
before the guided upsampling. On this stack the mask covers 64 % of the
first background region, 92 % of the second, 9 % of the fan and 26 % of the
ghost region (`flat.f32` is written beside `depth.f32`). The seams went,
but Travis found the background still "blobby": the depth now ramped
smoothly from frame 0 at the matrix to the far end, and the ramp walks
through frames 20-60.

**What the background is, frame by frame.** Warping every frame onto the
first background region and comparing (`bgframes_a.png`): the matrix at
the lower left is sharp at frame 0; in frames 20-60 it is a set of defocus
rings spreading over the background; from frame 80 on the region is a
featureless haze. The reference B in that region is closest to frame 0
(luma RMSE 0.031 against 0.046 for 0.6.3's B, 0.08 for frame 161), and its
depth map says why: the whole area within ~250 px of the matrix is at
frame 0-20, then it jumps within ~50 px to ~150 and stays there. Intermediate
frames are the worst choice for a defocused background and the reference
never shows them. Why our depth stage cannot find this on its own: in the
background the fine focus measure (6 px) is sensor noise, flat across the
stack to within 5 % (its argmax is the reference frame, the only unresampled
one), so the end priors and the regional vote decide, patch by patch. A
coarser measure (|G3 - G12| of the luma, averaged over 128 px) does carry
the signal: in the band and in the far background alike it is highest at
frame 0 (the matrix's haze) and falls monotonically to a floor by frame
~100.

Three variants of the depth stage, judged on the 1:1 sheets
(`report_real/h10_*`, `h11_*`, `h12_*`, `h13_*`):

| flat pixels' depth | far background | band beside the matrix | fan | notes |
|---|---|---|---|---|
| 0.6.3: SGM, then masked blur | smooth ramp, no seams | rings (frames 20-60) | clean | blobby |
| data term = least fine-scale energy (`--flat-min`) | smooth | rings | **blotch back** (needle gaps are "flat") | the fine measure is noise; settled on intermediate frames (median 55) |
| nearest *textured* pixel's depth, far end beyond | smooth | wrong (the matrix's haze counts as textured and carries far frames) | clean | dark cut-out around the matrix |
| **nearest *sharp* detail's depth, far end beyond (0.6.4)** | smooth, like the reference | the matrix's frame, soft ramp | clean | no rings, no seams |

The 0.6.4 rule, applied at depth resolution after the SGM, to every pixel
without confident sharp detail within the dark reach (the confidence
dilated by the reach; `wc` is the curve's contrast whatever the peak's
position, so end-framed objects count): the band value is the confident
pixels' depth spread by pull-push over an area pyramid, weighted by the
dilated confidence Gaussian-spread over 1.5 x the flat width and x4 (about
1 for 100 px, 0.5 at 190 px, gone by 400 px); beyond the band the pixel
takes the quieter end of the stack, the end whose coarse structure (the
luma volume's first and last 8 frames, |G3 - G12|, averaged over the flat
width) is the weaker, which on this stack is the far end everywhere; the
target is blurred over a quarter of the flat width so band and far end
meet gradually. Sharp detail keeps its SGM depth; the 1/4 reference keeps
its bytes. Cost: the depth stage is unchanged at ~3.0 s (the volume stage's
slice blur 2.0 -> 2.7 s remains).

Per-region scores (luma RMSE against the reference, tone-matched / edges;
depth RMSE in frames):

| run | plates | fan | crystal edge | bg edge | matrix flat |
|---|---|---|---|---|---|
| 1/4 reference (`--b-reference 1`) | .0412 / .0766, 8.1 | .0299 / .0648, 12.2 | .0398 / .0879, 19.2 | .0657 / .1213, 3.4 | .0459 / .0804, 5.0 |
| 1/2 + reach + vote | .0412 / .0762, 20.5 | .0299 / .0648, 13.5 | .0394 / .0872, 19.8 | .0661 / .1213, 7.6 | .0458 / .0803, 7.1 |
| 0.6.3 (slice + depth blur) | .0412 / .0759, 13.7 | .0297 / .0649, 12.6 | .0397 / .0875, 24.1 | .0657 / .1217, 8.2 | .0458 / .0803, 6.4 |
| least fine-scale energy | .0434 / .0761, 29.4 | .0538 / .0755, 53.1 | .0396 / .0852, 33.2 | .0667 / .1219, 16.5 | .0454 / .0796, 17.7 |
| nearest textured, far end | .0412 / .0758, 14.4 | .0306 / .0650, 17.2 | .0414 / .0878, 52.3 | .0658 / .1218, 9.9 | .0463 / .0800, 19.5 |
| **0.6.4 (nearest sharp, far end)** | .0412 / .0760, 14.2 | .0295 / .0645, 13.0 | .0397 / .0876, 21.9 | .0664 / .1213, 8.6 | .0458 / .0803, 7.8 |

Luma is within noise except where a variant broke something (the fan with
the least-energy term); the metric cannot see seams or rings, the sheets
can. What 0.6.4 does not reproduce: the reference keeps more of the
near-frame bokeh (the dark shadow of the crystal in the second background
region, the haze blobs right of the matrix); ours renders those areas from
the far end, flatter but equally smooth.

0.6.4's B defaults: volume at 1/2, dark prior 1 with reach 24, end vote 48,
flat smoothing 128 / reach 24 with the band-and-far rule, P1 0.06 / P2 0.8
in effect, guided radius 8. `--b-reference 1` (1/4 volume, dark prior 1, no
reach, no vote, no flat smoothing, unscaled P1/P2) is the 0.5.x B, still
byte-identical to 0.5.5 (checked after every change).

## 35. Which B parameters matter: a one-at-a-time sweep (2026-11-07, night)

`stackcli sweep` (0.6.6: one alignment, one pass 1 + volume per group of
pass-1 settings, then a B per variant at ~7 s) over every depth / composition
parameter, each on both sides of its default, base = the GUI's defaults
(`--c-radius 32`), 162 frames at full resolution. Two scores per variant:
the share of B's pixels whose luma differs from the base by more than 2 % of
full scale (what the two-up viewer's Difference overlay shows at its default
level), and `compare_real.py` against Helicon's B R9 S1 (tone-matched luma
RMSE on edges and depth RMSE in frames, mean of the five regions).

| variant | pixels changed > 2 % | 99.9th pct | depth RMSE vs Helicon (base 19.9) |
|---|---|---|---|
| depth volume 1/4 | 7.9 % | 14.5 % | 17.6 |
| flat blur 0 (background smoothing off) | 6.9 % | 13.5 % | 21.5 |
| confidence ramp 0.55..0.75 | 4.7 % | 9.9 % | 21.3 |
| flat blur 256 | 3.5 % | 8.6 % | 19.8 |
| confidence ramp 0.35..0.55 | 2.5 % | 7.3 % | 18.8 |
| B radius 32 | 2.1 % | 7.4 % | 21.1 |
| aggregation 0 / 64 | 2.1 / 1.6 % | 8.2 / 7.7 % | 20.5 / 19.4 |
| dark reach 48 / 0 | 1.5 / 1.4 % | 6.1 / 6.5 % | 19.7 / 23.4 |
| depth focus window 12 / 3 | 1.2 / 0.5 % | 6.0 / 3.4 % | 20.2 / 20.0 |
| halo prior 0 / 4 | 0.9 / 0.5 % | 5.8 / 3.9 % | 19.4 / 20.6 |
| fine-focus window 5 / 15 | 0.7 / 0.7 % | 4.1 / 4.3 % | 20.0 / 19.9 |
| C radius 0 / 64 | 0.6 / 0.4 % | 4.8 / 5.1 % | 20.4 / 19.9 |
| dark prior 0 / 2 | 0.5 / 0.03 % | 4.1 / 1.3 % | 18.9 / 20.2 |
| fine-focus range 24 / 6 | 0.5 / 0.3 % | 4.3 / 3.3 % | 20.1 / 19.9 |
| SGM P2 0.8 / 0.2 | 0.08 / 0.05 % | 1.9 / 1.4 % | 19.8 / 20.0 |
| soft label 1 / 6 px | 0.08 / 0.07 % | 1.9 / 1.8 % | 19.9 / 19.9 |
| end vote 0 / 96 | 0.03 / 0.00 % | 0.8 / 0.4 % | 19.9 / 19.9 |
| SGM P1 0.015 / 0.06 | 0.00 / 0.00 % | 0.7 / 0.7 % | 19.9 / 20.0 |

What it says:
- **Only a handful of parameters change B visibly**: the volume's resolution,
  the flat-region (background) smoothing, the confidence ramp, the
  aggregation width, the dark prior's reach and the halo prior. Everything
  else, SGM P1 / P2 above all, moves well under 1 % of the pixels: the
  smoothness penalties are not what decides B at these settings; the priors,
  the guided upsampling and the refinement are. Smoothing is a C control and
  never touches B (the composite is written as is); the C radius reaches B
  only through the guide and moves a pixel in a few hundred.
- **The tone-matched luma RMSE against Helicon is useless for ranking**: every
  variant lands within ±0.0002 of 0.1386. The depth RMSE moves (1/4 volume
  −2.3 frames, dark reach 0 +3.5), but at ~20 frames it measures the broad
  disagreement with Helicon's map (§13), not quality. Ranking stays with the
  eye at 1:1: the two-up viewer sorts variants by their visible change and
  tints the differing pixels.
- Hence the GUI's **Run sweep** presets (0.6.8): quick = background blur off,
  confidence ramp low / high, 1/4 volume, dark reach 0 (6 B's, ~1.5 min);
  thorough adds background blur 256, aggregation 0 / 64, dark reach 48, halo
  prior off, B radius 32 (12 B's, ~2.5 min). The sweep records each
  variant's change share in `sweep.json` (`change_pct`).
- **Travis's verdict on the thorough set** (11/07, night, at 1:1 in the
  two-up viewer): background blur 256 is the best of the twelve. The 1/4
  volume looked good at first and was then rejected for its doubled plate
  edge (§32 again, even with the 0.6.2–0.6.4 additions on top). The others
  "sacrifice blur in parts where there should be features, or create a really
  ugly background". Next: blur 192 / 256 / 384 / 512 and the flat reach at
  12 / 48, all on the 1/2 volume.

## 36. Background blur vs features: the band decoupled, weak features kept (2026-11-07, night)

Follow-up sweeps on the 1/2 volume (`Q:\_sweep_bg`, `Q:\_sweep_keep`).
Blur 192 / 256 / 384 / 512 change 1.8 / 3.5 / 5.4 / 6.8 % of B's pixels
(the change grows almost linearly with the width); blur 256 with flat reach
12 / 48: 4.1 / 3.8 %. Travis: **512 gives the best background but is among
the worst for detail** in parts that should have features.

Why one number does both: `flat_blur_px` sets the slice blur in flat pixels,
the coarse-energy averaging, the band's reach (1.5 x) and the band-to-far
transition (0.25 x); and every pixel farther than `dark_reach_px` (24) from
*confident* detail is overridden by the fill, whatever focus evidence it has.
At 512 a sharp object's frame reaches 768 px and weak features far from
strong detail take it.

Two controls, CLI only (`--flat-band W`, `--flat-keep S`; DepthParams
`flat_band_px`, `flat_keep`): the band width decoupled from the blur width,
and the override scaled by `1 - S min(contrast / c_lo, 1)` so a pixel with
focus evidence of its own keeps that share of its SGM depth. Change vs base:
512 alone 7.4 %; 512 + band 128: 2.9 % (so most of 512's effect was the
band's reach, not the smoothing); 512 + keep 1: 8.3 %; 512 + band 128 +
keep 1 / 0.5: 5.1 / 3.0 %; 256 + keep 1: 5.1 %.

**Travis: none of them is good.** Helicon "still handles things much better";
the failures to fix are banding and "waxy lines" in the background while
features keep their detail. Parked in favour of the retouch tool; the next
step here is another black-box study of Helicon's background (§31) aimed at
those two artefacts. The two controls stay (default off, bit-identical B).

## 37. End frames, Helicon's background depth, and raw vs denoised frames (2026-11-08)

**a. The SGM argmin never returned an end frame (fixed, 0.6.10).** `k_argmin`
and the prototype's `argmin_subpixel` clamped the winning label to `1 .. K-2`
before the parabola and returned the clamped label even when the parabola was
skipped, so a pixel sharpest in frame 0 or K-1 was rendered from frame 1 or
K-2. Now an end-frame winner returns the frame itself (no parabola). On the
real stack (denoised frames, same transforms) the depth changed on 9.9 % of
pixels, by 1.00 frame on average (max 25 through the fill's spread); 14 % of
the crystal_edge region now sits at frame 0 (Helicon: 8 %, 0.6.9: 0.2 %) and
8 % of blurbg_a (Helicon: 20 %). B differs by more than 512 LSB16 on 1.1 % of
pixels, all in those end-frame areas; at 1:1 the crops are indistinguishable.
Sheets: `helicon_B_analysis\report_argmin`.

**b. What Helicon's B does in the defocused background** (its `_dmap`, 162
grey levels = integer frames, against ours; `bg_depth_stats.py`,
`report_bg\bgdepth_*.png`). Helicon does not send the background to one end
of the stack. Its depth there is a smooth, blob-shaped field that spans the
whole range and follows the coarse content: the crystal's shadow in blurbg_b
is at frames 0-30 in Helicon, at 156 in our 0.6.4 band fill (xl_h11_band);
the upper left of blurbg_a is frame 0 in Helicon, the far end in ours. Within
a blob Helicon is near one end (black or white in the overview), and the
transitions between blobs are wide and continuous. Its map is 2-3x smoother
at 4 px (RMS Laplacian) than ours in every region, but it still has 20-33 %
step pixels in the blur regions, so smoothness is not the difference: the
choice of frame per blob is. Our fill renders the whole region as a monotone
ramp from one end over ~200 px, which is the flat "wax" and, where the ramp
crosses many frames, the faint concentric lines Travis sees.

**c. A model of it** (`bg_helicon_model.py`: all 163 frames warped onto the
regions at 1/4, band-pass energies |G(a) - G(b)| per frame, per-pixel
argmax / centroid, smoothed plainly or weighted by the peak energy, scored
against Helicon's map by Spearman rank correlation). Best candidates:

| region | best candidate | Spearman | RMSE fr | ours (band fill) |
|---|---|---|---|---|
| blurbg_a | bp 2-8 px argmax, energy-weighted smooth 32 px | 0.88 | 42 | 0.71 / 49 |
| ghost | bp 3-12 px, window 24, weighted smooth 32 px | 0.96 | 8.5 | 0.89 / 13.9 |
| blurbg_b | bp 2-8 px argmax, weighted smooth 96 px | 0.73 | 57 | **0.77** / 86 |

The band-pass argmax has Helicon's shape (same blobs, same side of the stack)
but stays mid-grey where Helicon snaps to the ends: Helicon's background is
closer to "each blob takes the end its coarse evidence points to, with wide
soft transitions" than to a literal smoothed argmax. The near-frame shadow in
blurbg_b is predicted by no band-pass scale up to 24-96 px. Not adopted yet;
the next step is to try that rule in the fill (per-blob end from the
band-pass argmax instead of one global end vote) and judge at 1:1.

**d. Raw vs denoised frames.** Travis denoises every frame in Camera Raw
before stacking (Helicon saw the same denoised frames, so all comparisons so
far are fair). The same 163 shots exported without denoise
(`Q:\163 noise`, noise MAD 0.0071 vs 0.0028 at level 0, 2.5x) stacked with the
same transforms: the depth map differs by 20.7 frames RMS (49 % of pixels by
4 or more frames), B's luma by 0.0167 RMSE. At 1:1 the raw-frame B loses half
the fan's needles to a yellow blotch (the 0.6.4 fill: with `--b-reference 1`
the needles survive apart from a small blotch at the base) and the plates'
internal striations go soft in both B variants. So B's confidence thresholds
and the fill's "sharp detail" gate are tuned to the denoised noise level and
are not noise-adaptive; the measured `noise_mad` is not used by the depth
stage. Open item: scale `c_lo`/`c_hi` (and the fill's gate) by the measured
noise, then re-check both frame sets. Sheets: `report_noise\*_1to1.png`.

## 38. Noise-adaptive confidence, and a race that made the depth map differ run to run (2026-11-08)

**a. The race (fixed, 0.6.11).** Two stacks of the same frames with the same
settings gave different depth maps: 9-16 % of the pixels by 2-3 frames on
average, up to 100, and B differed on half the pixels (1 % by more than
512 LSB16). Pass 1's dumps (`fold --volume`, now also `--cache 1`) were
bit-identical run to run, with or without the frame cache, but `flat.f32`
was not: the flat mask's two Gaussian kernels were built, used and freed in
sequence on one stream, the second built at the address the first had just
freed while the first blur could still be running, so the first blur read the
second kernel's taps. It only showed with two streams per GPU and the frame
cache (the timing that left the blur in flight); one stream, or `--cache 0`,
always gave the same (correct) answer. The kernels now live until the stage's
sync. Verified: one GPU, two GPUs, one or two streams, cache on or off all
give the same hashes for flat, depth and B. The argmin numbers of §37a stand
(the two runs compared there happened to land in the correct state).
*Correction (§43, 0.6.20): the mechanism above was wrong and the fix only
moved the timing. On the 201-frame stack B still differed run to run, also
with one GPU and one stream; the real cause is the taps' upload itself.*

**b. Why B degrades on raw frames (§37d).** The confidence `(vmax - vmed) /
vmax` of a focus curve is an SNR test: the curve's median in a flat or
defocused pixel *is* the noise floor of the measure (median N / vmed = 0.92
to 1.01 in every region tested), so the ramp 0.45 .. 0.65 means a peak 1.5 to
2.7 times the noise. Raw frames have 2.1x the floor (0.0593 against 0.0279 for
the denoised set, measured the same way), and the same detail falls under
the ramp. A per-pixel reformulation does not recover it: floor-subtracted
contrasts, absolute signal thresholds, SNR gates and their products
(`noise_conf.py`, 16 candidates) reproduce the denoised set's confidence map
on raw frames no better than mean |dw| 0.07-0.16, because the per-pixel
estimate of the signal is itself noisier.

**c. What works: scale the ramp by the stack's floor.** The floor is measured
once per stack from the two middle frames (their difference / sqrt 2 through
the slice's own chain, normalised by the first frame's luma, box-aggregated,
square-rooted; median over the depth grid: `depth_noise_floor`), together
with the pyramid MADs the finalize already used (now taken before the depth
stage, `NoiseEstimate`). The ramp is multiplied by
`(conf_noise_ref / floor)^conf_noise_gamma`, never above 1, at least 0.5,
and exactly 1 within 5 % of the reference (so the calibration stack is
bit-identical: verified). Defaults: ref 0.028 (this stack's denoised floor),
gamma 1. On the raw set: gamma 0.5 (ramp 0.31 .. 0.45) shrinks the fan's
blotch to its base; gamma 1 (0.22 .. 0.32, the clamp) brings the needles
back with a small dark blob at the base, and the plates' striations return
(the denoised set remains better). The raw set's depth percentiles move back
onto the denoised set's (plates p50/p95 68/109 -> 66/100, denoised 67/105;
blurbg_a p50 88 -> 33, denoised 46). More noise flips pass in flat areas
(4+ frame steps 10 -> 12 % in plates), which the fill absorbs. The fill's
"sharp detail" gate uses the same ramp, so it benefits too. Sheets:
`report_noise\fan_gamma_1to1.png`, `plates_gamma_1to1.png`. Left open: C's
smoothing gate already adapts (it reads the MADs); the SGM penalties do not.

## 39. The warp's resampling filter as an option (2026-11-08)

Travis asked for Helicon's Preferences choice of interpolation ("to improve
the preservation of details"). `Grid::interp` / `--interp` / the render
panel's "Interpolation": bilinear, bicubic (a = -0.75, the reference; the
only one with the OpenCV 4 `quantize` path), Lanczos 3 (6 x 6 taps), Lanczos
8 (16 x 16 taps), all at exact double positions with reflect borders, weights
normalised. The default is unchanged and bit-identical (hashes of §38).
On the 163-frame stack, B and C with smoothing 2 / C radius 32:

| filter | pass 1 stream | pass 3 | total | B luma RMSE vs bicubic | high-pass energy, fan / plates |
|---|---|---|---|---|---|
| bilinear | 3.85 s | 1.41 s | 15.2 s | 0.0056 | 0.915 / 0.918 |
| bicubic | 3.96 s | 1.41 s | 15.3 s | 0 | 1 |
| Lanczos 3 | 4.06 s | 1.52 s | 15.6 s | 0.0014 | 1.012 / 1.014 |
| Lanczos 8 | 5.17 s | 2.41 s | 18.6 s | 0.0016 | 1.028 / 1.031 |

So Lanczos 8 keeps about 3 % more fine-detail energy than bicubic for 3.3 s
more per render, Lanczos 3 about 1 % for 0.3 s; bilinear loses 8 %. At 1:1
the fan and plates crops are hard to tell apart by eye
(`report_noise\fan_interp_1to1.png`, `plates_interp_1to1.png`); the two-up
viewer's Difference overlay shows where they differ. Every output changes
with the filter (the alignment does not: it has its own B-spline sampler).

## 40. The background takes the frame where its own structure peaks (2026-11-08, 0.6.16)

The follow-up to §37 b-c. Beyond the band beside sharp detail, 0.6.4 sent
every flat pixel to the *quiet* end of the stack (the end frame with the
lower coarse |G3 - G12| energy): a monotone ramp from one end, the flat
wax and the concentric lines Travis saw. Three alternatives, all behind
`--flat-end` (DepthParams::flat_end):

- **loud** (1): the end frame with the *higher* coarse energy, a soft
  transition where the two are within `--flat-end-soft` of each other.
  Picks the near end almost everywhere in the blur regions (blurbg_a p50
  2, blurbg_b p50 1 against Helicon's 13 and 29): the end frames alone
  do not know where the far matrix is in focus. Rejected.
- **argmax** (2): per pixel, the frame where the focus measure V (the
  aggregated slice, pooled over 24 px) peaks over the whole stack, then an
  energy-weighted 32 px blur (the §37c model, which scored Spearman 0.88 /
  0.96 against Helicon's map). A first version measured |G3 - G12| of the
  luma volume instead and failed the same way as loud (p95 41-52 where
  Helicon has 155): that volume is blurred over 12 px, which removes the
  2-8 px band the far matrix lives in. Built before `k_cost` overwrites V;
  0.35 s per render.
- **snap** (3): the argmax field pushed to the ends over a transition of
  `--flat-end-soft`. Faint arcs where blobs meet. Rejected.

Depth percentiles p5 / p50 / p95 against Helicon's map:

| region | Helicon | quiet (0.6.15) | argmax |
|---|---|---|---|
| blurbg_a | 0 / 13 / 155 | 0 / 51 / 144 | 0 / 13 / 103 |
| blurbg_b | 1 / 29 / 155 | 15 / 150 / 162 | 3 / 19 / 158 |

At 1:1 (`report_loud\blurbg_*_V_1to1.png`) the argmax background shows
the crystal's dark shadow and the warm haze blobs where Helicon shows
them; the quiet one is a flat brown wax with a brightness step. The change
is confined to the flat mask: 45 % of the flat pixels move (mean 51
frames), 0.7 % outside it; ghost, plates, matrix and crystal_edge
statistics are unchanged. **argmax is the default from 0.6.16**;
`--flat-end quiet` is the 0.6.4 .. 0.6.15 depth-stage background,
bit-identical to 0.6.15's depth map (the 0.6.15 B also needs `--flat-blend 0`
since §41; 0.6.4's B differs further by §37a's end-frame argmin and §38's
ramp scaling, which sit under every mode).
Still open: Helicon reaches the far end more often in blurbg_a (p95 155
vs 103), and its map carries frame-0 speckle inside far regions that ours
does not reproduce (and need not).

## 41. The rings in a flat background come from the two-frame blend, not from the depth (2026-11-08, 0.6.18)

A new 201-frame stack (IMG_5535 .. 5735) rendered with 0.6.16 against Helicon's
B R9 S1 and C S1 (`helicon_B_analysis\compare_stack.py`, the general
comparison: regions picked by detail, tone-matched luma RMSE, 1:1 sheets in
`report_newstack`). Whole image: B RMSE 0.036 with 1.04x Helicon's fine
detail, C RMSE 0.041 with 0.97x. In the flattest tile (a smooth yellow
gradient at 4950, 3900) our B showed faint concentric rings where Helicon is
smooth. The depth there sweeps the whole stack (p5 7 .. p95 200, 199
distinct frames): the §40 field ramps where there is no evidence. The rings
are not the ramp itself but how the composition renders it: in regions the
crisp map calls soft, B is the smooth-depth path, a blend of the two frames
on either side of Ds at each pixel, and along a ramp neighbouring pixels blend
different pairs, so every integer crossing of Ds draws a contour.

Fix: widen that blend in flat regions. `k_labels` takes a per-pixel width
map (1 + flat x (W - 1), the flat mask resized to the output grid), the
weight becomes `clip(1 - |Ds - k| / w, 0, 1)`, and `compose_frames` admits
every frame within W of Ds. Outside the flat mask w = 1: the reference
composition, bit-identical. `--flat-blend W` (StackOptions::flat_blend, 0 =
off). On the new stack, the flat tile's 4-12 px band energy (Helicon 0.0032):

| blend | 0 | 8 | 16 | 32 |
|---|---|---|---|---|
| band RMS 4-12 px | 0.0026 | 0.0024 | 0.0023 | 0.0020 |
| band RMS 12-40 px | 0.0071 | 0.0069 | 0.0067 | 0.0062 |

The numbers move little because Helicon's own bokeh energy is larger than the
rings; the 1:1 sheet (`flat1_blend_1to1.png`) is decisive: 8 frames already
removes every ring, 16 and 32 look alike and nearer Helicon's soft blobs.
Detail tiles are untouched (mean change under 1e-4). Default 16 from 0.6.18;
compose time 2.5 -> 2.9 s because every frame now carries weight somewhere.

A property to keep in mind, not changed: the window is clipped at the
stack's ends, so a flat pixel whose depth is exactly frame 0 takes weights
1, 15/16 .. 1/16 over frames 0 .. 15, whose centroid is frame 5. "The quiet
end" in a flat region is therefore a 16-frame mixture leaning 5 frames
inward, not the end frame; with argmax (the default) the field rarely sits
on an end, so this shows mainly with `--flat-end quiet` or `snap`.

## 42. Review of 0.6.10 .. 0.6.18 (2026-11-08, 0.6.19)

Four read-only reviewers over the day's commits (core depth / compose, warp
and file naming, GUI, FINDINGS against the code), every finding checked in
the source. Fixed in 0.6.19, none of which changes a default render:

- The sweep's pass-1 grouping key (`pass1_key`) did not include the warp's
  filter, so `--interp` variants in one grid shared the first variant's pass
  1 (volume, guide, C) and sweep.json reported them as unchanged. No sweep
  with `--interp` was run before the fix.
- The two-up viewer kept the previous render's B and C under the plain names
  after a render into another output folder (the outputs arrive with the
  folder's B.tif path for retouching), and its folder-qualified fallback
  renamed before checking for the same file, so repeat adds duplicated the
  entry. In-memory render outputs now replace by name whatever their folder;
  only files from different folders coexist.
- `--b-reference 1` had drifted from "the 0.5.x B exactly": §38's ramp
  scaling applied to it, and §37a's end-frame argmin. It now sets
  `conf_noise_gamma 0` and clamps the argmin's end frames to 1 / K-2 as the
  prototype (`StackOptions::sgm_end_frames`, `DepthParams::end_frames`).
- The ramp's `c_hi` was clipped at 0.95 even when the noise scale was exactly
  1; the scaling now applies only when the scale is below 1.
- `warp` and `fold` take `--interp`, so a Lanczos stack and its fold check
  share a warp; the end-frame energies are not computed for argmax / snap.
- Text: the "bit-identical" claims above and in the CLI help and headers
  were narrowed to what holds; the argmax field was described in three
  places as the rejected |G3 - G12| luma version; the Lanczos tooltip said
  "some 10 s more" against §39's measured 3.3 s; the Flat smoothing tooltip
  and README described the quiet-end default.

Unsupported rather than wrong, and still open: §40's case for argmax rests on
agreement with Helicon's depth map and on 1:1 crops matching Helicon's; no
measurement from the source frames shows the far matrix sharpest at the
chosen frames (the §25 rule).

## 43. The flat mask's race, found for real: a synchronous cudaMemcpy that is not (2026-11-08, 0.6.20)

Checking 0.6.19 against 0.6.18 on the 201-frame stack at full resolution, B
differed (11.4 M of 91 M samples, peak 0.44 of full scale) while C was
byte-identical. Two runs of the same build differed as much (9.5 M samples,
peak 0.24), so did two runs of 0.6.18, with one GPU, and with one stream per
GPU; `--flat-blur 0` gave identical B and depth every time. `flat.f32`
differed on 47 % of its pixels (mean 0.05, max 0.62), and the hashes of flat
and B repeated from a small set of states across all those runs: not noise,
a race with a few outcomes. With `STACKER_FLAT_DEBUG=DIR` (pass 1 snapshots
the mask's five stages on the stream and writes them after the stage) two
runs were identical at every stage, in the state the lucky runs had shared:
the one extra `cudaMalloc` before the chain hid it.

The cause is in `up()` (imgops.cu), which puts a Gaussian kernel's taps on
the device with `cudaMalloc` + `cudaMemcpy`. The Runtime API's "API
synchronization behavior" says of the synchronous copy: *for transfers from
pageable host memory to device memory, the function returns once the
pageable buffer has been copied to the staging memory for DMA transfer, but
the DMA to the final destination may not have completed.* The DMA is on the
legacy default stream, which blocking streams wait for and non-blocking
streams do not. Pass 1 runs on `cudaStreamNonBlocking` streams and the flat
mask builds its two kernels immediately before the blurs, so the blur could
read taps that had not landed; the fusion's stream is a blocking one, which
is why the depth stage's kernels (built the same way) never showed it, and
§38's rearrangement only changed the timing. `up()` (and the same helper in
compose.cu, and the alignment's twiddle upload) now end with
`cudaStreamSynchronize(0)`, which waits for the staged DMA. The resampler's
coefficient tables went through the same helper.

Verified on the 201-frame stack at full resolution: two default runs, one
GPU, and one stream per GPU give one hash each for flat, depth, B and C
(flat c02d7116, B 06a4010d), the state the debug runs and the lucky runs had
shared. That B is also the hash one 0.6.18 run produced, so 0.6.19's code
changes left the default render bit-identical, as §42 claimed. A render is
also faster now (22 s against 38 s for the stack command with B and C),
presumably because the corrupt masks marked more of the image flat and the
16-frame blend then drew every frame into the composition there; not
measured separately.

## 44. The B parameter search, round 1: scored against the source frames (2026-11-08, night)

Travis judged 0.6.20's B at 1:1 ("looks great") and opened the parameter search
he had planned: the 201-frame stack only, his eye at 1:1 as the judge, with an
objective helper that measures against the source frames rather than against
Helicon (his 11/06 rule: another stacker agreeing proves nothing). Winners
become the defaults; the knobs stay in Advanced.

**The helper** (`helicon_B_analysis\score_sweep.py`): per pixel, the most
fine-detail energy any source frame carries there is the *oracle*: every frame
is warped onto the output grid with the stacker's own centred similarity
(`affine_for`, centre (W−1)/2; the middle frame lands within 0.5 px of the
base B on every tile), its high-pass (σ 1.5) is squared and pooled over σ 4,
and the maximum over the stack is kept. A B at the oracle kept all the detail
a frame had; above it, it carries energy no frame has (doubled edges, halos,
seams); below it, it lost detail. Per 900 × 600 tile: *retention* = median of
E_B / oracle over the textured pixels (oracle above its 70th percentile), its
10th percentile (the worst tenth), and *excess* = the share above 1.2 × the
oracle. Six detail tiles (the most high-pass energy in the base B, non-
overlapping) and two flat tiles, where the 4..40 px band energy relative to
the base shows rings and bokeh. The score rewards noise as well as detail
(one frame's noise is in the oracle, a blend averages it away), so it ranks,
it does not decide.

**Round 1** (`Q:\_sweep1`, 47 variants, 15 min): one knob at a time, both
sides of its 0.6.20 default, base = the GUI's defaults (`--c-radius 32`).
Full-resolution depth volume (`--depth-fraction 1`) runs the 3080 Ti out of
memory and is off the table. The base B retains 0.868 of the oracle (median),
0.617 at the worst tenth, excess 0.001.

| variant | B pixels changed | retention | worst tenth | excess | flat band vs base |
|---|---|---|---|---|---|
| soft labels 1 px (default 3) | 1.1 % | **0.970** | **0.764** | 0.024 | 1.00 |
| aggregation 0 px (default 32) | 3.7 % | 0.873 | 0.640 | 0.001 | 1.01 / 0.96 |
| refine range 24 (default 12) | 2.1 % | 0.873 | 0.634 | 0.001 | 1.00 |
| depth volume 1/4 (default 1/2) | 5.5 % | 0.870 | 0.618 | 0.001 | 0.92 / **1.24** |
| base | 0 | 0.868 | 0.617 | 0.001 | 1.00 |
| every flat-region knob, the priors, SGM, the ramp, end vote, focus radius, interpolation | 0 .. 20 % | 0.868 ± 0.001 | 0.617 ± 0.003 | 0.001 | see below |
| B radius 32 | 2.3 % | 0.864 | 0.580 | 0.001 | 1.00 |
| refine range 6 | 2.2 % | 0.858 | 0.569 | 0.000 | 1.00 |
| aggregation 64 px | 4.0 % | 0.855 | 0.540 | 0.001 | 1.0 |
| refine window 15 / 5 px (default 9) | 4.4 % | 0.852 / 0.837 | 0.52 / 0.62 | 0.000 | 1.00 |
| soft labels 6 px | 1.1 % | 0.726 | 0.463 | 0.000 | 1.00 |

Background, the two flat tiles' 4..40 px band energy against the base's (1 =
the same): `--flat-end quiet` 0.51 / 0.94 (the smoothest, and what Travis
rejected as wax in §36), flat blur 512: 0.84 / 0.80, flat blur 256: 0.83 /
0.90, flat blend 32: 0.94 / 0.99, dark reach 48: 0.98 / 1.03; flat blur 0:
1.15 / 1.32, flat blend 0: 1.03 / 1.02, flat keep 1: 1.04 / 1.08, flat min 1:
0.99 / 0.65 (and 20 % of B changed), depth volume 1/4: 0.92 / 1.24. The
confidence ramp, the priors and SGM do not move either kind of tile by more
than 0.3 %, as §35 found.

What it says:
- **Detail in B is decided by the composition, not the depth.** Only the
  soft-label width, the refinement's range and window, the aggregation width
  and the B radius move the detail tiles; every depth knob, including the
  ones §35 ranked high (volume resolution, the ramp, the priors), leaves the
  retention within 0.001. The depth map puts the right frame under the
  detail; how many frames are blended there is what sharpens or softens it.
- **Soft labels at 1 px gain 10 % of the fine detail and 15 % at the worst
  tenth, and are visibly crisper at 1:1**, but the excess share rises to 2.4 %
  and the sheets show why: hard label edges become seams (a jagged step across
  the crystal in detail6). 6 px blends a quarter of the detail away. The
  default 3 is on the soft side; 1.5 .. 2.5 are round 2's first candidates.
- **Refine range 24 and aggregation 0 gain a little each** (0.005) without
  seams; aggregation 0 shows a blotch in detail2 that needs Travis's eye.
- **The background knobs are orthogonal to detail** (the flat mask keeps
  them off textured pixels, as designed), so background smoothness can be
  tuned on its own: flat blur 256 and flat blend 32 are the calm candidates
  that do not go all the way to the waxy quiet end.

**Round 2** (`Q:\_sweep2`, 20 variants, 5 min): soft labels 1.5 / 2 / 2.5,
refine range 18 / 24 / 36, aggregation 0 / 16, their combinations, flat blur
192 / 256 / 384, flat blend 24 / 48, and a combined candidate.

| variant | B pixels changed | retention | worst tenth | excess |
|---|---|---|---|---|
| soft 1.5 + refine 24 | 2.7 % | 0.946 | 0.735 | 0.009 |
| soft 1.5 | 0.5 % | 0.944 | 0.721 | 0.009 |
| soft 2 + refine 24 + aggregation 0 / 16 | 4.3 / 3.5 % | 0.923 / 0.922 | 0.707 / 0.705 | 0.004 |
| soft 2 + refine 36 | 3.3 % | 0.922 | 0.701 | 0.004 |
| soft 2 + refine 24 + flat blur 256 + flat blend 32 | 5.1 % | 0.921 | 0.698 | 0.004 |
| soft 2 | 0.1 % | 0.918 | 0.683 | 0.004 |
| soft 2.5 | 0.0 % | 0.893 | 0.648 | 0.002 |
| refine 36 / 24 / 18 | 3.1 / 2.1 / 1.2 % | 0.873 / 0.873 / 0.871 | 0.64 / 0.63 / 0.63 | 0.001 |
| aggregation 0 / 16 | 3.7 / 2.2 % | 0.873 / 0.872 | 0.64 / 0.64 | 0.001 |
| base (soft 3, refine 12, aggregation 32) | 0 | 0.868 | 0.617 | 0.001 |

Background band energy against the base: flat blur 256 + flat blend 32:
0.79 / 0.90; flat blur 256 alone 0.83 / 0.90; 192: 0.88 / 0.93; 384: 0.83 /
0.88; flat blend 48 alone 0.86 / 0.98, 24: 0.98 / 0.99.

What it says:
- **The soft-label width is a clean dial** between detail and seams:
  retention 0.970 / 0.946 / 0.918 / 0.893 / 0.868 at 1 / 1.5 / 2 / 2.5 / 3 px,
  excess 0.024 / 0.009 / 0.004 / 0.002 / 0.001. At 2 px the sheets show
  crisper crystal faces and plate striations than the base with no seam I
  can find; at 1.5 the step across the crystal in detail6 is still faintly
  there. 2 px is the recommendation, 1.5 the aggressive option.
- **Refine range 24 adds a little on top** (+0.004 at soft 1.5, +0.005 at
  soft 2) without a visible cost; 36 adds nothing more. The refine window
  stays at 9 (both sides lose).
- **Aggregation 0 and 16 are out**: the gain is within noise and both draw a
  vertical blotch across the plate in detail2 that the base does not have.
- **Background**: flat blur 256 with flat blend 32 is the calm candidate
  (a fifth less ring energy, detail tiles untouched); 384 and 512 buy little
  more. These are Travis's to judge at 1:1, as blur 512 was in §36.

Candidates for Travis's eye (Compare > Open sweep folder on `Q:\_sweep1` and
`Q:\_sweep2`, Difference overlay on): `soft-px-2_refine-r-24_vol-agg-px-16` is
not it (aggregation); the detail candidate is **soft 2 + refine 24**, seen
in the combined `flat-blur-256_flat-blend-32_soft-px-2_refine-r-24` with the
background change and in `soft-px-2_refine-r-36` without it; the aggressive
one is `soft-px-1.5_refine-r-24`. Nothing is changed in the defaults until he
has looked.

**Travis's verdict** (the same night, at 1:1): the combined candidate "is
quite good, it has some merits over regular base B. I prefer how it handles
the background. But base B is still better for details." So **0.6.22 makes
flat blur 256 and flat blend 32 the defaults and leaves soft labels at 3 and
the refine range at 12**. The lesson for the helper: fine-detail energy
against the sharpest frame ranks the variants, but a higher score is not a
better picture. Soft labels 2 measured +6 % and looked worse to him; the
blend's smoothing of frame-to-frame noise and rims is part of what reads as
quality. The score stays a way to find candidates and to catch losses (soft 6,
the refine window, aggregation 64), not a judge. The Run sweep presets are
rebuilt from this table (soft labels 2 / 1.5 / 4, refine range 24, the 0.6.21
background, blur 512, blend 64 / 0, the quiet end, refine window 5 / 15, B
radius 32, 1/4 volume); the 11/07 set (ramp, dark reach, halo prior,
aggregation) is gone, being within noise.

## Open items

- **B: doubled plate edge / glow** (§25a): the C-style aggregation does not
  work on the normalised focus measure; needs rim rejection or ownership.
- **C: stacked rim outlines inside bright plates** (§25): present at any C
  radius and in Helicon's C too. Same mechanism as the band outside the edge
  but with no stronger neighbour to own the pixel.
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
- **DNG input** (Travis, 11/08): the reader takes uncompressed 8/16-bit TIFF only; a DNG (a raw mosaic, or
  Camera Raw's enhanced linear raw) is lossless-JPEG compressed and in camera colour before white balance, so
  reading it means a raw developer stage (LibRaw), not a reader tweak. The Camera Raw 16-bit TIFF export stays
  the path for now.
- **AutoFOV link**: bring AutoFOV's per-frame blur values (and focus position)
  into the frame list next to the alignment wobble, from a capture-side
  sidecar, so frames can be culled or weighted from capture data too.
