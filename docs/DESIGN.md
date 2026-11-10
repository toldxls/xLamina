# Focus Stacker — Design (draft 1, 2026-11-05)

Working name: **Stacker**. A Helicon-class focus-stacking app for Windows that
uses every NVIDIA GPU in the machine and **reports the image geometry
precisely**, so AutoFOV's µm/px calibration still applies to the stacked result.

## Decisions so far

| Topic | Decision |
|---|---|
| Platform | Windows PC, native app |
| GPUs | 2× RTX 3060 Ti 8 GB, 2× RTX 3070 Ti 8 GB, 1× RTX 3080 Ti 12 GB. All Ampere sm_86; none has NVLink |
| Typical job | 100–300 frames. Frames are 6960 × 4640 (~32 MP, taken from the 2026_08_15 Mitutoyo set; camera model to confirm) |
| Input / output | 16-bit TIFF + JPEG in → 16-bit TIFF (+ JPEG) out. **No DNG** (TIFF input is already rendered) |
| Optics | Camera + infinity objective + tube lens on a rail (near-telecentric) |
| Methods (v1) | A weighted average, B depth map, C pyramid, + retouching |
| UI | Desktop GUI, Qt 6 |
| Retouch | Brush from a source frame, brush between methods, auto halo/noise hints. *Non-destructive project files: not chosen* |
| Fusion R&D | Hybrid luma/chroma, depth-guided pyramid, noise-aware selection |
| Geometry | Crop policy **selectable per job**; every output carries a geometry report |
| AutoFOV link | Pull calibration (GET /calib) → metadata; optional burned-in scale bar |
| Dev | Claude Code runs directly on the Windows GPU PC, in its own repo |

## Toolchain

- C++20, MSVC 2022, CUDA 12.x/13.x, fatbin for sm_86 / sm_89 / sm_120 + PTX
- CMake + vcpkg: Qt 6 (Widgets), libtiff, libjpeg-turbo, lcms2 (ICC), Exiv2 (XMP), nlohmann-json, cuFFT
- No OpenCV in the core (its CUDA modules need a custom build); write our own kernels

### Repo layout
```
core/     stackcore — C++/CUDA engine, no Qt dependency
cli/      stackcli  — headless runner: batch jobs, benchmarks, tests
gui/      stackgui  — Qt 6 client of core
ref/      CPU reference implementations (double precision, slow, obviously correct)
tests/    synthetic-stack generator, GPU-vs-ref, 1-GPU-vs-N-GPU determinism
tools/    comparison harness (Helicon vs ours: difference images, noise, sharpness)
```

## Memory and I/O budget (why it streams)

- One frame, 32 MP RGB: 194 MB as 16-bit (= FP16, the GPU storage format; math is FP32), 388 MB in FP32
- 200-frame stack: **39 GB** of 16-bit TIFF on disk, so frames stream through the GPUs and the stack never sits in VRAM
- Pyramid partial result per GPU: ~0.7–1 GB (FP32 coefficients + uint16 source index). Working set ~2–3 GB, which fits the 8 GB cards
- **The bottleneck is probably not the GPU:**
  - Disk: 39 GB at 3–7 GB/s NVMe is 6–13 s; from a NAS or HDD it takes minutes.
  - Compressed TIFF (ZIP/LZW) decodes at only ~100–300 MB/s per core, so either use many decode threads or export uncompressed.
  - PCIe: consumer boards often split lanes x8/x8, or put a card on a chipset x4 link. Measure, don't assume.
- **Two-pass pipeline** (alignment pass, then a fusion pass).
- **Superseded (FINDINGS 26 and 52):** the app now keeps the frames resident in its own pinned cache, and a fast NVMe SSD feeds it as fast as a RAM disk does, so a RAM disk is not needed. The planning notes below are kept as written.
- **Workflow at the time: a RAM disk on a 192 GB machine.** TIFFs already sit on a RAM-file-system, and today a single 3080 Ti stacks them in Helicon. That changes the picture:
  - **The disk bottleneck is gone.** Reads run at memory speed, so the two-pass pipeline's second read is nearly free, and the app needs **no cache of its own** (that would just double RAM use). The RAM disk *is* the cache: the app streams files into a small ring of pinned buffers.
  - A 39 GB stack on the RAM disk + pinned buffers + app working memory fits easily in 192 GB, even with two stacks staged at once.
  - What's left to measure in M0:
    - the RAM-disk driver's real read rate (some Windows RAM-disk drivers are much slower than raw memcpy)
    - the TIFF decode rate (uncompressed TIFF is ~memcpy; ZIP/LZW needs many threads)
    - host memory bandwidth: each byte crosses RAM ~3× (RAM disk → pinned → DMA), and a dual-channel desktop has ~60–90 GB/s in total
  - **Observed: the RAM disk gave Helicon little speedup.** So Helicon is limited by its own processing, not I/O. Two reasons the disk wasn't the limit:
    - Windows' file cache already holds recently read/written TIFFs in the 192 GB anyway.
    - An NVMe read of a 39 GB stack (~6–13 s) is a small share of Helicon's total time.

    This is good news: compute is what multi-GPU replaces. The RAM disk (or a warm file cache) only starts to matter once our per-frame time gets near the read rate.
  - **Baseline to beat: Helicon + 3080 Ti, 200 frames in 10–15 s** ≈ 50–75 ms/frame ≈ **3–4 GB/s** end-to-end. Profile it once with Task Manager (GPU "Cuda/Compute" graph, CPU per-core) to see where its time goes.

### Speed targets (200 frames, 32 MP)
| Config | Target | Reasoning |
|---|---|---|
| Ours, 3080 Ti only | ≤ 10 s | sanity check: must match Helicon on the same card |
| 3080 Ti + 3070 Ti | ~7–9 s | ~1.7× on the parallel part, plus ~1–2 s of work that doesn't split |
| All five cards | **~3–5 s** | ~3.3× on the parallel part; that fixed work (pair chaining, merge, output TIFF write) becomes the floor |
| Broadcast: A + B + C + variants | ≈ the time of one render | the bigger day-to-day win: method comparison is ~36 s+ in Helicon today |

- Requirement this implies: 3–4× faster means **~12–15 GB/s of sustained input**. Every stage must scale, not only the GPU work.
- That needs **uncompressed TIFF** (or many decode threads) and enough host memory bandwidth (dual-channel ~60–90 GB/s, shared by ~3 copies). The RAM disk becomes useful at this rate even though it wasn't for Helicon.
- M0 measures every stage against these targets before any hardware is bought.

## Pipeline

```
ingest (CPU thread pool: TIFF/JPEG decode → ICC → pinned host buffers)
  → PASS 1  align      (luma only, cheap)         → transform table + geometry report
  → PASS 2  warp+fuse  (full RGB, streamed)       → partial results per GPU
  → merge partials → post-process → collapse/compose → crop policy → write TIFF/JPEG + XMP + sidecar
```

### 1. Alignment
- Match each adjacent pair (i, i+1) independently, so the pairs spread across GPUs:
  - coarse: phase correlation on a downsampled luma image (cuFFT)
  - fine: ECC / Lucas–Kanade refinement of a **similarity** transform (tx, ty, scale, rotation), plus an optional brightness gain to correct exposure flicker
- The CPU chains the pair transforms into one transform per frame relative to the reference frame.
- Robustness: a low correlation peak marks a pair as bad. Retry against frame i+2 and flag it in the report.
- Scale is **always estimated and reported**; a "lock scale" option forces s = 1 in the warp. With your near-telecentric objectives the expected change is tiny. The report will show whether that holds; a wrong tube lens breaks telecentricity.
- Warping is on the fly in pass 2 (texture-based bicubic/Lanczos). Warped frames are never stored.

### 2. Geometry report — closes the AutoFOV gap
Every job writes `<name>.stack.json` and XMP fields in the output TIFF:
- per-frame transform table (tx, ty, s, θ, gain, match quality)
- the reference frame and its native size; the output size in px; crop margins per edge (L/R/T/B px)
- the output scale relative to the reference frame, and the **effective µm/px and FOV in mm** (from the AutoFOV calibration)
- a coverage map (how many frames cover each output pixel), exportable as an image
- which crop policy was used

Crop policies (chosen per job):
1. **Native-crop**: the reference frame's pixel grid, cropped to the area every frame covers. µm/px equals the calibration exactly.
2. **Native-full**: the reference grid at full sensor size; borders filled from whichever frames cover them. The coverage map shows where depth of field is reduced.
3. **Rescaled**: crop, then resample back to the input pixel size. The width stays constant in pixels, and the reported factor says exactly how much µm/px changed.

The reference frame can be first, middle or last (default: middle, which minimizes the largest warp).
For why this matters and what it reveals about the optics, see **Key investigation** below.

### 3. Fusion: frame-parallel with a merge that doesn't depend on order
Every method folds frames into a partial result that can be merged in any order. Ties break on frame index, so **output is identical whatever the GPU count or mix**, and that is a test invariant.

- **A, weighted average**: w = FM^p; accumulate Σw·I and Σw. Accumulate in FP64 or fixed point to keep the result deterministic.
- **B, depth map**: focus measure FM (e.g. a modified Laplacian or local variance at scale σ). Keep the best FM and its frame index per pixel. Then smooth the index map with an edge-aware filter. Compose with interpolation between neighbouring frames, which takes one more streaming pass over the frames. B also outputs the depth map itself.
- **C, pyramid**: build a Laplacian pyramid per frame. At each level, keep the coefficient with the highest score, where the score is |c| plus local energy over 3×3 (Burt–Adelson salience), and record its source index. Merge the partial pyramids, apply consistency filtering, then collapse.

#### Attacking your pain points
- **Pyramid noise.** Per coefficient, plain max selection over N noisy frames has expected noise magnitude that grows like σ·√(2 ln N), which is why noise climbs with frame count. *Noise-aware selection*: estimate σ per pyramid level from the stack (robust MAD over flat regions). A coefficient wins on |c| only when it clears k·σ. Below that, take the coefficient from the depth-map-guided frame.
- **Colour/contrast shift**, via *hybrid luma/chroma*: pyramid-fuse the luminance only, and take chroma from the smoothed depth-map frame.
- **Halos**, via the *depth-guided pyramid*: where C's chosen index disagrees with B's by more than k frames and the coefficient isn't strongly dominant, fall back to B's frame. Halos are partly unavoidable (the background behind a blurred foreground edge is never cleanly seen in any frame), so the remainder goes to retouch hints.

### 4. Multi-GPU scheduler
- At startup, enumerate the devices and microbenchmark each one (pyramid build of one frame) to get throughput weights.
- Work goes out in dynamic frame chunks through a queue that idle GPUs pull from, so mixed cards balance automatically with no static split.
- Each GPU runs 2–3 CUDA streams so upload, compute and download overlap; host buffers are pinned.
- Three modes:
  - **split**: frames divided across GPUs, one render
  - **broadcast**: every frame goes to every GPU, and each runs a different method or parameter set
  - **batch**: whole stacks per GPU

  See the Hardware plan section for why broadcast matters.
- **Windows gotchas:**
  - GeForce cards run under WDDM: there is no TCC mode, and peer-to-peer GPU transfer is generally unavailable, so merges go through host RAM. That's fine: it's ~1 GB per GPU, once per job.
  - The 2 s TDR watchdog applies to **every** WDDM GPU, not only the display card. Keep every kernel ≪ 100 ms (tile the big ones).

## Key investigation: reference frame, scale, and the optical train

This is a main goal of the project, not just a stacking detail. The alignment
pass measures the optical system on every stack, for free.

### What the reference frame is
Each frame sees the specimen from a slightly different position along the rail.
To fuse them, the same feature has to land on the same pixel in every frame, so
they must share one pixel grid. The **reference frame** is the one whose grid
everyone uses. It's copied through untouched, and every other frame is resampled
onto it. Its pixel size (the calibrated µm/px) becomes the output's pixel size.

### Why frames differ in scale even though magnification "doesn't change"
Camera and objective move together, so the *plane in focus* always has the same
magnification. That part is correct. Scale change comes from **perspective**: a
feature that's out of focus in frame i and sharp in frame j moved closer to the
lens in between. Unless the optics are perfectly object-space telecentric
(entrance pupil at infinity), its image grows slightly.

- Relative scale change across the stack: **ε ≈ total rail travel / entrance-pupil distance**
- A perfect telecentric objective has ε = 0. A reversed or macro lens with its pupil ~50 mm away gives ε ≈ 2% per mm of travel.
- Mitutoyo-style infinity objectives are *designed* near-telecentric, but the *as-built train* may not be. Candidate culprits, to test rather than assume:
  - tube-lens choice and spacing
  - bellows extension
  - a vignetting element acting as the stop
  - the bellows-lock slop already suspected for the lens decenter

### How big the warp is
The correction is a similarity transform: shift + one uniform scale s = 1 + ε + a tiny rotation.
Shapes do **not** bend, stretch or shear. Displacement is zero at the scale
centre and largest at the corners. The half-diagonal of a 6960 × 4640 frame is
~4180 px:

| ε across the stack | Corner movement | Typical cause |
|---|---|---|
| 0.05% | ~2 px | good telecentric train, shallow subject |
| 0.2% | ~8 px | telecentric objective, imperfect train |
| 1% | ~42 px | non-telecentric, ~0.5 mm of travel |
| 2% | ~84 px | reversed/macro lens, ~1 mm of travel |

**Resampling cost:** every non-reference frame needs a sub-pixel *shift* anyway
(rail wobble, stage drift), so every frame is interpolated whatever the scale.
With Lanczos or bicubic, a 0.1% zoom costs no more than a 0.3 px shift. If the
measured ε is below the noise floor, **lock scale** keeps the shift and drops the scale term.

### What the reference choice changes: crop, not sharpness
- **Frames shrunk to fit the reference** no longer cover the canvas → empty borders → crop.
- **Frames enlarged to fit** overfill the canvas → nothing is lost inside it.

| Reference | Effect |
|---|---|
| Most zoomed-in frame | All others are enlarged → scale causes no crop (only drift trims edges) |
| Least zoomed-in frame | All others shrink → maximum crop |
| Middle frame (default) | Half the stack goes each way → intermediate crop; the largest single warp is halved |

This is the "same input width, different output pixel count" effect. The
geometry report shows it per edge, per policy.

### The measurement catch: there's no single true µm/px unless ε ≈ 0
After warping, each region of the output is drawn at the *reference frame's*
projection. Features at the reference focal depth are exact. Features at other
depths are off by their share of ε, so the µm/px error across the subject is
**±ε × (fraction of the travel away from the reference depth)**. The report gives a
**µm/px range across the stack depth**, not just one number, and flags when the
range exceeds a threshold (e.g. 0.1%). When ε ≈ 0 the range collapses and the
AutoFOV calibration holds everywhere in the image. That is the goal for the
optical train.

### Prototype findings (11/5): read docs/FINDINGS.md
- **Specimen relief biases scale and drift.** Defocused light from other depths
  makes intensity alignment read a 3D specimen's *shape* as magnification
  (−50% on a synthetic dome) or tilt (2× on a tilted plane). It scales with
  NA² × slope and swamps a telecentricity residual.
  - Optics diagnostics must come from a **flat target**; that path is validated: ε to 2.4%, tilt within 2σ, crop exact.
  - Stacking with telecentric optics should likely **lock scale**.
- **Sub-pixel interpolation precision matters.** OpenCV's 1/32 px warp grid
  biased scale ~20%. CUDA texture units (1/256 px weights) can't be used for
  alignment; interpolate in software.

### Optical-train diagnostics (by-product of alignment)
The alignment produces per-frame (tx, ty, s, θ). Against rail position z
(frame index × step size), fit:

| Fit | Meaning | What it optimizes |
|---|---|---|
| ds/dz (%/mm) | telecentricity residual. The sign says whether the pupil is in front of or behind the object (entocentric vs hypercentric) | tube lens / spacing / bellows extension |
| d(tx,ty)/dz | lateral image drift per mm = **angle between rail and optical axis**, atan(drift ÷ travel) | rail and camera squareness, bellows lock |
| scale fixed point (cx, cy) | the **focus of expansion** (FOE): the rail direction's projection. It equals the optical axis only if the rail is parallel to it, and is defined only when ε ≫ noise. *(Corrected 11/5: draft 1 called it the optical axis.)* | combined with the lens-centering axis (−3.75%), FOE − axis gives rail tilt |
| dθ/dz | roll per mm | rail twist |
| residual jitter of tx/ty about the fit | step-to-step repeatability; vibration | rail mechanics; links to the AutoFOV vibration/settle data |
| residual maps after the similarity fit | effects a similarity can't model (distortion changing with focus) | whether a richer model is ever needed |

Tracking over time:
- Each job appends its diagnostics to a **history log** (CSV/SQLite), keyed by objective, tube lens, bellows setting, lock state and date.
- That log is the A/B tool: change one thing (lock vs unlocked, tube-lens spacing), shoot, and compare the trend lines.
- Test protocol: ordinary specimen stacks already measure everything above. A dedicated target (tilted stage micrometer, textured across its depth) gives the cleanest numbers.
- To convert frames to µm of travel, the step size is needed (entered per job, or read from the rail/AutoFOV if available).

## Hardware plan: 2× 3060 Ti, 2× 3070 Ti, 1× 3080 Ti

| Card | VRAM | Bandwidth | Board power | Share of a frame split* |
|---|---|---|---|---|
| RTX 3080 Ti | 12 GB | 912 GB/s | 350 W | ~30% |
| RTX 3070 Ti ×2 | 8 GB | 608 GB/s | 290 W | ~20% each |
| RTX 3060 Ti ×2 | 8 GB | 448 GB/s | 200 W | ~15% each |

\*Stacking kernels are memory-bound, so throughput should roughly track
bandwidth: the five cards together come to about 3.3× one 3080 Ti. The real
shares come from the M0 benchmark; the dynamic queue balances them automatically.

### The honest bottleneck estimate
- Back of envelope: one 32 MP frame through warp + focus measure + pyramid is maybe ~10 GB of memory traffic, so ~15 ms on the 3080 Ti and ~30 ms on a 3060 Ti. **M0 measures the real number.**
- At that rate the five cards could eat ~150+ frames/s, which is ~30 GB/s of decoded TIFF.
  - From NVMe (~7 GB/s) the disk feeds only one or two GPUs.
  - **From the existing RAM disk, the supply is plausibly 10–20+ GB/s (M0 measures it), which is enough to keep ~3–5 cards busy in split mode.** The real limits then become host memory bandwidth and the PCIe lanes each card gets. That is why the slot layout matters.
- Where more GPUs *do* pay off:
  1. **Broadcast mode, "one read, many renders":** each frame is read once and sent to every GPU, and each GPU renders a different method or parameter set. A, B, C, the hybrid and noise-aware variants, or a parameter sweep for the comparison harness all finish in the time of one read. Depth-guided C and hybrid luma/chroma *need* B and C anyway, so they can run on separate cards at the same time.
  2. **Heavier, better algorithms:** multi-scale focus measures, the depth-map interpolation pass, full-resolution alignment refinement, and retouch re-renders.
  3. **RAM-cached re-renders:** after the first read, a re-render with new parameters is limited by RAM speed, not disk.
  4. **Batch:** whole stacks per GPU.

### PCIe: how much link each card needs
- Split mode at ~30 frames/s × 194 MB ≈ 6 GB/s per card → **PCIe 4.0 x4 is workable, x8 is comfortable**.
- Broadcast mode sends each card the full input stream (≤ disk rate, ~7 GB/s), so the same rule holds.
- **Avoid x1 mining risers** (~1 GB/s Gen3): each card would spend ~0.2 s per frame just uploading.
- There's no P2P/NVLink on these cards under Windows, so merges go through host RAM. That's fine at ~1 GB per card per job.

### Build options
| Option | What | Verdict |
|---|---|---|
| **1. Current desktop (192 GB, RAM disk), 2 cards. CONFIRMED** | 3080 Ti + a 3070 Ti in the second PCIe 5.0 x16 slot (power: OK). Both cards are Gen4, so the links run at 4.0. On consumer boards a second "x16" slot is usually **electrically x8** (CPU lanes split x8/x8: ~14 GB/s each, plenty) or **x4 through the chipset** (~7 GB/s, shared with NVMe/USB: still workable for split mode). Check the "Bus Interface" field in GPU-Z, e.g. `PCIe x16 4.0 @ x8 4.0` | **Start here** for M0–M3. Target ~7–9 s per 200 frames. Measure before buying anything |
| 2. Consumer board + risers | CPU x8/x8 plus chipset slots; every chipset slot shares one ~x4 Gen4 uplink | Cheap; fine for split mode on 3–4 cards, chokes broadcast mode |
| 3. **Used EPYC open-frame rig** | single-socket EPYC 7002/7003 (128 PCIe 4.0 lanes, 16–32 cores for TIFF decode) on a 7-slot x16 board (e.g. ASRock Rack ROMED8-2T) + 128–256 GB 8-channel RDIMM + Gen4 x16 riser cables | **Best fit for all 5**: full links, enough RAM to cache whole stacks, cores to decode. Build it only if M0 numbers justify it |
| 4. Threadripper (TRX50) | new HEDT | Same benefits as 3 at a much higher price |
| 5. Several PCs over LAN | the order-independent merge makes this possible | Later at most; needs 10 GbE + shared storage |

### Power and physical
- GPU board power totals ~1.33 kW, plus CPU/board ~250 W, so **~1.6 kW** with everything flat out. Ampere cards (3070 Ti/3080 Ti especially) also throw brief transient spikes well above their rating.
- A 120 V / 15 A circuit allows ~1.44 kW continuous. So:
  - **Power-limit the cards to ~60–70%.** Memory-bound work typically loses little; M0 measures it.
  - Use one 1600 W ATX 3.x PSU or two PSUs; ideally a 20 A or 240 V circuit.
- Five 2–3-slot cards don't fit a normal case: use an open frame with Gen4-rated x16 riser cables and good airflow.
- **TDR caveat (correction to draft 1):** Windows' 2 s GPU-timeout watchdog applies to *every* GeForce card under WDDM, not just the one driving the display. All kernels must be short (tiled). TdrDelay can be raised in the registry as a backstop.
- Display: let a 3060 Ti drive the monitor. The 3080 Ti keeps its 12 GB for the viewer/retouch cache and the largest share of compute.

## GUI (Qt 6 Widgets)
- Job list; source-frame strip with a scrubber; parameters per method; render queue.
- GPU-backed zoomable viewer (CUDA–OpenGL interop) with overlays: the coverage map, crop rectangles for all three policies, frame footprints, the depth map, and halo/noise hint maps.
- A **geometry panel** that shows the report live: px in, px out, crop per edge, µm/px, FOV.
- Progressive preview: the merge works frame by frame, so the result builds visibly while it renders.

### Retouch
- Brush from a **source frame**: a slider picks frame k, with a live preview of aligned frame k under the cursor. Frames are loaded from the RAM cache or re-read and warped on demand, with neighbouring frames prefetched.
- Brush between **methods**: paint B or A over C, and vice versa.
- **Auto hints**:
  - halo map: C's index disagrees with B's, near strong occlusion edges
  - noise map: coefficients selected below the noise threshold

  Clicking a hint suggests a source.
- Undo stack in memory. Strokes are kept as a list internally, so project files can be added later cheaply (not in scope now).

## AutoFOV integration
- **Pull calibration**: GET /calib needs a session token, so the stacker implements the device's POST /login challenge-response with the device password. Find the device by mDNS (Windows 10+ resolves .local natively) or a manual IP. If the device is offline, enter µm/px manually or use a lens preset.
- The µm/px goes into the XMP + sidecar and feeds the geometry report.
- **Capture record from AutoFOV**: this solves calibration binding (which calibration applies to *this* stack). The firmware's `wifiNotifyStackComplete()` already sends a WebSocket `stackDone` event, captured at the moment the stack completes, with:
  - `fov`: FOV in mm at capture
  - `obj`: objective index
  - `stepMm`: step size. **Units caveat:** the key says mm, but the value is `stackStepSize`, which the firmware and TFT treat as **µm** (`STEP_TABLE` 0.156–3.593, `BASE_STEP_UM` 0.1562). The stacker reads it as µm.
  - `imgs`: frame count; `durS`: stack duration; `dist`, `err`: TOF distance and error
  - `vst`: vibration state, plus the per-stack V/H blur figures

  The device keeps no history of past stacks (`/vibhist` covers only the last one). So the stacker keeps a **WebSocket connection to AutoFOV during capture sessions** and logs every `stackDone` with a timestamp to a local capture log. On import, the TIFF folder is matched to a log entry by frame count + EXIF capture times (`DateTimeOriginal` survives a Lightroom TIFF export: first/last frame vs event time − `durS`). The user confirms the match.
  - From the matched entry: µm/px = FOV ÷ frame width, and z per frame = index × step. Both feed the geometry report and the optical diagnostics. Vibration state goes in the history log too, which can explain alignment jitter.
  - Fallbacks: if the stacker wasn't running during capture, type the values in. A small firmware-side stack-history ring + `GET` endpoint is an *option*, not yet decided.
- **Scale bar**: optional, burned into a *copy* for export only (the master TIFF stays clean). Length follows a 1-2-5 sequence; position and style are configurable.

## Verification
- A CPU reference for every method, with GPU output matching within tolerance; 1-GPU and N-GPU runs give identical results.
- **Synthetic stacks** with ground truth: a known height-map scene plus texture, defocus PSF per frame, and a known shift/scale per frame. These measure alignment accuracy, **check the geometry report against the truth**, and score depth-map accuracy.
- Real stacks: your TIFF sets with Helicon outputs alongside. The harness produces difference images, noise σ in flat regions, and edge sharpness/MTF.
- Benchmarks: frames/s per GPU, and an I/O vs decode vs compute vs merge breakdown.

## Milestones
| # | Deliverable |
|---|---|
| M0 | Toolchain builds on Windows (CMake/vcpkg/CUDA/Qt); GPU enumeration. **Benchmarks that decide the hardware:** per-frame compute time per card, at stock and power-limited; PCIe H2D bandwidth per slot; TIFF decode rate per core; disk rate |
| **M1** | **Alignment + geometry report + optical-train diagnostics (CLI)**: ds/dz, drift angle, scale fixed point, jitter, plus the history log. Fills the AutoFOV crop/scale gap and starts optimizing the train *before* any fusion exists. Run it over existing stacks (2026_08_15 etc.) on day one |
| M2 | Methods B and A, multi-GPU scheduler, CPU reference, determinism tests, CLI end-to-end |
| M3 | Method C + noise-aware + depth-guided + hybrid luma/chroma; Helicon comparison harness |
| M4 | Qt GUI: jobs, viewer, parameters, progressive preview, overlays, geometry panel |
| M5 | Retouch: brushes from source frame and between methods, hint maps |
| M6 | AutoFOV calibration pull, scale bar, XMP/sidecar polish, installer |

## Open questions
1. ~~GPU models / current PC~~ answered: 3080 Ti + 3070 Ti in the current desktop. Remaining: the electrical lane width of the second slot (GPU-Z), and PSU wattage. ~640 W of GPUs + CPU, with Ampere transient spikes, wants ≥ 1000 W.
2. ~~System RAM~~ answered: 192 GB, with TIFFs on a RAM disk (which RAM-disk software? Its read speed matters).
2a. Camera model: are all stacks 6960 × 4640 (32 MP)?
2b. ~~Rail step~~ answered: known, and **AutoFOV already broadcasts it**. See "Capture record from AutoFOV" below.
2c. Optical-train variables to log: objective(s), tube lens, bellows extension, lock state.
2d. Power: is the machine on a 15 A or 20 A circuit?
3. ~~Storage / compression / baseline~~ answered: RAM disk; exports are **uncompressed** (~189,000 KB each in Explorer = 6960 × 4640 × 16-bit RGB exactly); Helicon baseline is 10–15 s per 200 frames. Decoding is just a copy, so the CPU stays free to feed the GPUs.
4. **Calibration binding**: proposed answer is the `stackDone` capture log (above). Is the stacker PC on the LAN with AutoFOV during capture, and do you want the firmware-side stack history as a backup?
5. Colour space of the TIFF exports (ProPhoto / Adobe RGB / sRGB)? Fusion working space: gamma-encoded vs linear is an R&D item for M3.
6. Do you have stacks with matching Helicon outputs (and known Helicon settings) to use as a benchmark?
7. Repo: a private GitHub repo under toldxls? Name?
