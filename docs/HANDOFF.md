# Handoff: Mac prototype → Windows CUDA (2026-11-06)

## 1. Where things stand

| Area | Status |
|---|---|
| Alignment (pairs + exact global solve) | Done, validated on synthetic ground truth (§1–2) |
| Geometry report (crop per edge, µm/px over depth, 3 crop policies) | Done. Explains Helicon's crop on the real stack (§6) |
| Optical-train diagnostics (telecentricity, tilt, FOE, jitter, lead-screw) | Done; valid on FLAT targets only (§3) |
| `compare`: measure Helicon's output geometry | Done (§6) |
| Fusion A / C / noise-aware Cn / variants | Done; order-independent folds, bit-identical across worker splits |
| **B (depth map)** | Close to Helicon B, still being improved (§8–14) |
| B↔C slider (`blend_bc`) | Done (§10) |
| CUDA / Qt / multi-GPU | **Not started.** That's this phase |

### B pipeline (current defaults, `fusion.run(depth_method="sgm")`)
1. Pass 1 fold: A, B0, C pyramid (per worker, merged in any order).
2. Focus + brightness volume at 1/4 sensor resolution (normalised Laplacian,
   `depth_fm_px` 6). Cached as `fmvol_q_*.npz` for fast experiments.
3. Volume aggregated edge-aware over 32 px (`vol_agg_px`), against ghost lines (§14).
4. Cost:
   - focus term, weighted by focus-curve contrast (ramp 0.45 → 0.65)
   - darkness prior for featureless pixels (§9)
   - end-frame prior for out-of-range regions (§12)
   - spill-zone darkness preference beside EXTENDED bright objects (found by
     morphological opening, 80 px reach, §13)
5. SGM, 8 directions (P1 0.03, P2 0.4, edge-aware), sub-frame argmin, guided
   upsampling.
6. Refinement: fine focus (9 px window) within ±12 frames, only where confident
   and not in a spill zone.
7. Composition:
   - narrow soft labels (3 px)
   - continuous smoothed depth (edge-aware) for blurry content
   - the two mixed by fine detail and focus confidence (§12)

### Baseline numbers (must reproduce on the PC)
`python tools/bench/score_b.py '{"default": {}}'` on the 2026-11-05 stack:
**RMSE all 0.0228, edge-band 0.0199, edge glare 2.3 %, depth~Helicon 46 %.**
Half-res fusion of 163 frames takes ~2.5 min on an M3 Pro CPU.

## 2. Open problems (priority order)

1. **B: ghosty glow along plate edges.** This is Travis's latest feedback
   (11/06), still visible after §14. What is known:
   - It is light from the defocused plate edge, or its doubled rim, in frames
     chosen for pixels near the edge.
   - Aggregating over 32 px helped (glare 3.1 → 2.3 %) but did not remove it.
   - Not yet tried:
     a. a brightness-excess penalty: pixels whose local brightness in frame k
        exceeds their minimum over frames *because of an adjacent brighter
        plate*;
     b. composition that never blends a plate-edge frame into its dark
        neighbours (the 3 px soft-label blur);
     c. the per-pixel *peak-to-neighbour* consistency check: does the pixel's
        chosen frame agree with the edge it sits next to?
   - Inspect with `roi_compare.py ... plates fan crystal_edge`.
2. **C / Cn: more bright glare spots than Helicon.**
   - Much of it is Helicon's tone curve (§10).
   - The sparkles exist in the source frames.
   - The coarse-from-B (Cg) and depth-locked (Cd) variants did not help.
3. **The scale-ramp question.** Shoot a FLAT textured target with the same setup
   (20×, 0.468 µm steps) to confirm the +2.5 % ramp is specimen bias. If so,
   default to `--lock-scale` for telecentric objectives.
4. A residual flat-target roll bias on far-defocused frames (§5 open items).

## 3. Moving to the PC

1. Unzip the handoff and clone the bundle:
   ```
   git clone Stacker.bundle Stacker
   cd Stacker
   ```
2. Python 3.12: `py -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`,
   then `python -m pytest -q tests`. Expect 13 passed, 1 xfailed.
3. Point at the data (PowerShell):
   ```
   $env:STACKER_FRAMES = "<RAM disk>\TIFF"     # the 163 IMG_*.tif + the three Helicon outputs
   $env:STACKER_OUT    = "<fast disk>\stacker_out\tiff_2026-11-05"
   ```
4. `python tools/bench/setup_stack.py` re-creates the analysis, fusion and
   Helicon prep (a few minutes).
5. `python tools/bench/score_b.py '{"default": {}}'` must print the baseline
   numbers above. That proves the PC reproduces the Mac.
6. CUDA toolchain:
   - Visual Studio 2022 (Desktop C++), CUDA 12.x (sm_86 for 3060 Ti / 3070 Ti /
     3080 Ti), CMake, vcpkg (libtiff, libjpeg-turbo, lcms2, nlohmann-json), Qt 6 later.
   - Check each card's PCIe link in GPU-Z ("Bus Interface").

## 4. Phase B: CUDA plan (start here on the PC)

M0, measure before optimising:
- per-card throughput of the core kernels (warp, focus measure, pyramid level);
- H2D bandwidth per slot;
- RAM-disk read rate;
- Helicon's time on this stack: B ~20 s, C ~9–10 s on the 3080 Ti (the target).

Port order. Each step is checked against the Python reference on the real
stack, within numeric tolerance:

| # | CUDA component | Python reference |
|---|---|---|
| 1 | **Done (§16)**: TIFF → pinned buffers → GPU streaming (thread pool, 2–3 streams/GPU) | `frames.read_rgb` (bit-exact), `fusion.load_aligned` |
| 2 | **Done (§17)**: similarity warp onto the output grid (software bicubic, 1.8 ms/frame) | `fusion.load_aligned` (1e-6 rms) |
| 3 | **Done (§18)**: pass-1 fold per GPU: A sums, B0 argmax, Laplacian pyramid C select; host merge. Full res 2.8 s on the 3080 Ti | `fusion.fold`, `fusion.merge` (byte-identical across GPU splits) |
| 4 | **Done (§19)**: focus + brightness volume at 1/4 res, guided-filter aggregation, in the pass-1 stream | `fusion._w_fmvol`, `sgm_from_volume` (aggregation part) |
| 5 | **Done (§20)**: cost volume, SGM (8 paths), sub-frame argmin, guided upsampling; 0.5 s on the 3080 Ti | `sgm_from_volume`, `sgm.py` (depth identical at 99.999 %) |
| 6 | **Done (§21)**: refinement pass + dual composition + finalize; `stackcli stack` writes all outputs. Full res 18.7 s | `fusion._w_refine`, `_w_b2dual`, `finalize` (end-to-end numeric check pending the Python reference run) |
| 7 | Alignment: exact cubic B-spline GN pairs (NOT texture interpolation) + host solve | `align.py` |
| 8 | Multi-GPU scheduler: split frames by measured throughput (3080 Ti + 3070 Ti first) | `fusion._chunks` + merge |

The fold/merge design is already order-independent: frames can go to any GPU,
and partial states merge on the host in any order with bit-identical B and C.

## 5. The test data

- Stack: 163 frames IMG_4923–IMG_5085.
  - Canon EOS M6 II, 6960 × 4640, 16-bit Adobe RGB TIFF, uncompressed.
  - 0.468 µm steps; higher frame numbers move the camera AWAY.
  - FOV 0.98 mm (0.1408 µm/px).
  - Specimen: yellow tyuyamunite plates on a brown matrix; defocused background on the right.
- Helicon outputs in the same folder: B (Radius 9, Smoothing 1), C (Smoothing 1),
  and Travis's B+C composite. All 6714 × 4404, referenced to frame 0.
- ROIs used in all comparisons are in `tools/bench/common.py`.

## 6. Suggested first message to Claude Code on the PC

> Read CLAUDE.md and docs/HANDOFF.md. Set up the Python environment, run the
> tests, run tools/bench/setup_stack.py with my TIFF folder at <path>, and
> confirm score_b.py reproduces the baseline. Then start Phase B M0: install
> the CUDA toolchain if needed and write the benchmark harness.
