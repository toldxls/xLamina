# xLamina

![xLamina](gui/icon/xlamina-128.png)

A focus stacker that uses one or more NVIDIA GPUs to stack frames quickly: 8/16-bit TIFF
(uncompressed, LZW or ZIP) or JPEG. It is a Windows program (C++ and Qt) built on CUDA.

Open a folder of frames, choose the output methods, press Render, watch the stacks build side
by side, and save. Every render also writes a geometry report: each frame's similarity
transform, the reference frame, the crop, and the effective µm/px over the stack depth when the
field of view and step size are known.

## Methods

| Output | What it does | Use it for |
|---|---|---|
| **B** | Depth map. Measures focus per frame, solves one smooth depth per pixel, then takes each pixel from the frame at that depth with soft transitions. | The cleanest tones and backgrounds. |
| **C** | Laplacian pyramid. Takes every detail, at every scale, from the frame where it is strongest. | The most detail and overlapping structures; noisier, and bright highlights can glow. |
| **Cn** | C where the detail is above the noise, B elsewhere. | C's detail without its grain in flat areas. |
| **CBn** | Cn that also falls back to B where C's choice disagrees with the depth map. | Fewer artefacts from frames far from the true depth. |
| **Cx** | CBn with a stricter test: a disputed choice stays only if the next finer scale chose the same frame. | Glow around defocused highlights. |
| **Cd** | Pyramid with its fine scales locked to B's depth. | Glints picked up from other depths. |
| **Cg**, **Cg2** | Pyramid with its coarsest scales (one more in Cg2) taken from B. | Glare from defocused highlights. |
| **H** | Brightness from CBn, colour from B. | C's detail with B's colour. |

Smoothing (C) sets how much of C's weak detail is taken from B: 0 is the raw selection, 3 is
Cn. Three more outputs sit under Tools > Developer outputs: A (focus-weighted average), B0 (B's
raw depth, unsmoothed) and CB (CBn without the noise rule).

With View > Developer controls on, the depth map is a tab beside the outputs and an entry in the
two-up viewer: the frame each pixel of B comes from, drawn in a palette (Grey, Viridis, Inferno,
Magma, Turbo) over a chosen range of frames, optionally in bands that repeat every few frames.
Brightness and contrast spread a few frames' difference over more colours, the frame under the
cursor is named beside it, and the two-up viewer can lay the map over another image.

## Hardware

64-bit Windows 10 (1903 or later) or 11 and an NVIDIA GPU: GeForce GTX 16 / RTX 20 series or
newer, with a current NVIDIA driver; a second GPU shares the work. It has been developed and
tested on one PC (RTX 3080 Ti and RTX 3070, 191 GB of RAM); smaller machines are handled by
falling back (fewer streams, the image rendered in strips, a coarser depth volume), which is
slower. A 200-frame stack of 31 Mpx frames wants about 6 GB of GPU memory for the default depth
resolution. At most 512 frames in one render.

The frames are read once and kept in memory for the rest of the job and for the next render of
the same folder, as long as they fit beside what the job itself needs (up to 60 % of the
machine's RAM): about 40 GB for 200 16-bit frames, half that for 8-bit. Frames that do not fit
are read from the drive again on every pass, which still works but runs at the drive's speed.
A fast SSD is enough; a RAM disk gives no extra speed.

## Install

Download `xLamina-<version>-win64.zip` from the Releases page, unzip it anywhere and run
`xLamina.exe` in the `xLamina` folder it holds. It needs a current NVIDIA driver (see Hardware).

The program is not signed, so Windows shows "Windows protected your PC" at the first start:
choose More info, then Run anyway (or, before unzipping, right-click the zip, Properties, tick
Unblock). A PC with Smart App Control switched on blocks unsigned programs outright.

When a render fails, the message says what happened and what to try. Help > Report a problem
saves one zip to the Desktop with the logs, the PC's details and the settings (no images, the
user folder's name replaced) to send with a description; the logs themselves are in
`%LOCALAPPDATA%\xLamina\logs`.

Updates: a few seconds after it starts, xLamina asks GitHub for the latest release (Help >
Check for updates at start turns this off). A newer one lights an **Update to x.y.z** button
at the right of the toolbar; it shows what changed and, on *Update and restart*, downloads
the zip, swaps the files in the folder xLamina runs from and restarts it. Settings are kept.
Help > Check for updates asks at any time.

## Use

Help > Manual (F1) opens the illustrated manual in the browser: every panel and control on annotated
screen grabs, with a search box. It is [docs/manual/index.html](docs/manual/index.html) in this repository and
`manual\index.html` in the release zip; `tools\manual.cmd <folder>` remakes its screen grabs from a render.

Open a folder (or drop one onto the window); File > Open files, or dropping several image files, stacks only
those files. Every TIFF or JPEG of the stack's image
size is a frame; other images in the folder are listed greyed out and left alone. Uncompressed
TIFF reads fastest. Compressed TIFF (LZW, ZIP, PackBits) and JPEG are decoded as they are read,
with the embedded profile kept. Camera raw files and DNG are not read: export TIFF from a
raw converter. Tick the outputs
you want, set the field of view if you know it (it gives the report its µm/px), press
Render. While it runs the viewer splits in two: on the left each frame as it is
processed (raw while reading, so the stack's wobble is visible, then warped onto the
output grid), on the right what is being built from them — the detail map through the
stack forwards, the depth map back through it, then the composition forwards. Zoom and
pan stay put as the frames change. Afterwards the tabs under the viewer switch between
the outputs at full resolution; clicking a frame in the list shows it at full
resolution. Live view sets the resolution of the live panes (each frame shown is copied
off the GPU, so Full slows the render). Save as… writes the current image as a 16-bit
TIFF, JPEG or PNG; the full set plus `report.json` is also written to
`<folder>_stacker`. The frames' ICC profile is kept in the outputs and used for display.

After a render each frame in the list shows its wobble (the shift in px and scale change
in % from the previous frame, from the alignment) and its sharpness (its focus energy
against the five frames around it, 1.00 = as expected). Frames whose shift is an outlier
are shown in red, frames whose focus energy fell under their neighbours' (blurred during
the exposure; under 0.90 by default, File > Preferences > General > Soft frame below) in
amber; Render > Drop shifted frames and Drop soft frames untick each kind, and unticked
frames are left out of the next render. A third line gives the frame's rolling-shutter
marks: "warp" is its displacement from band to band of sensor rows against its neighbours
(px), "band" the LED flicker's brightness ripple across its rows (% peak-to-peak). File >
Preferences > General > Correct rolling-shutter warp (checked by default: it removes the doubled
edges the row displacement causes) and Correct LED banding (unchecked by default) apply the
measured corrections to every frame; saved files get _rs / _band while a correction is on.
A frame warped far beyond the rest (shaken during its readout) is shown in violet and
Render > Drop warped frames unticks it.
The list is the stack order: Render > Sort frames by name or by time taken (the EXIF
capture time) reorder it, Tick all frames puts every frame back into the render.

Smoothing (C): 0 is the raw selection (in flat areas it picks the noisiest frame), 2 is about
Helicon's Smoothing 1, 3 is Cn.

File > Batch render… queues several folders (add them one by one, add every subfolder of a
folder, or drop several folders onto the window) and renders them one after the other with the
Render panel's current settings. Each folder's outputs go to `<folder>_stacker`; a folder that
fails or holds no stack is skipped and listed in the log at the end.

Add comparison image… (File menu, or drop a file onto a finished render) loads another
stacker's output as a tab beside xLamina's, so the two flip at the same zoom and position;
PgUp / PgDown step through the tabs.

Advanced (depth and composition), unticked by default, exposes the parameters of B's depth
pipeline and of the composition (depth resolution, focus window and power, depth window,
volume aggregation, refinement range and window, label softness, halo and dark priors, the
dark prior's reach, flat smoothing) at their defaults, for one-at-a-time experiments against
a 1:1 comparison. The defaults in short: depth is estimated at 1/2 of the sensor resolution
(1/4 left doubled outlines on bright edges); defocused background keeps the nearest sharp
detail's frame for about 256 px and beyond that follows its own focus peak with soft
transitions; flat regions blend up to 32 frames so a depth ramp draws no rings. The warp's
resampling filter is a choice (bicubic by default; Lanczos 3 or 8 for a little more fine
detail; bilinear). docs/FINDINGS.md has the evidence behind each default.

Cn is C at smoothing 3; with the panel's C smoothing at 2 the two differ on about 3 % of
pixels, so tick Cn only when C's smoothing is lower.

Performance holds the hardware settings: which GPUs, streams per GPU, reader threads,
alignment pairs in flight, and whether the frames stay resident. None of them changes the
result (B and C are bit-identical across all of them). Tools > Benchmark has three modes.
Quick (about 20 s) times up to 40 frames of the open stack at half resolution under the
current settings, each GPU alone, and with the frames streamed instead of resident, and can
apply the fastest. Thorough runs the whole stack with every setting varied in turn, then the
best of each combined. Single run renders the whole stack once on the current settings and
keeps each run in the table, to compare the machine against itself (an overclock, a driver).
A built-in synthetic stack stands in when no folder is open, or to compare machines. The
report can be saved or copied.

Live view costs a little speed (each image copied off a GPU costs its uploads about 3 ms,
so views are rate-limited); Off is the fastest setting.

All settings, the window layout, the theme (View > Theme: dark, light or the system style)
and the recent folders are kept in `%APPDATA%\xLamina\xLamina.ini`.

The same engine runs from the command line:

    stackcli run <folder> [<folder2> ...] [--out DIR] [--outputs B,C,Cn] [--fov-mm X] [--scale 0.5] [--interp bicubic] [--smoothing 2] [--cache 0] ...

## Build

Visual Studio 2022 (Build Tools with the C++ workload suffice), CUDA 13.x, and for the
GUI Qt 6.11 for MSVC 2022 under `C:\Qt\<version>\msvc2022_64` (the Qt online installer, or the
`qtbase` package from download.qt.io; only Core, Gui, Widgets and Network are used). The scripts take the
newest Qt 6 found there; set `QT_DIR` to choose another.

    tools\build.cmd        builds build\gui\xLamina.exe and build\cli\stackcli.exe
    tools\package.cmd      builds and zips a release into dist\

## Layout

    core/     stackcore, the C++/CUDA engine and the pipeline
    gui/      xLamina, the Qt application
    cli/      stackcli
    tools/    build and packaging scripts, the icon generator, the M0 hardware benchmark
    docs/     DESIGN.md and FINDINGS.md (the evidence behind every decision)

The engine was ported stage by stage from a Python reference implementation and checked
against it (bit-exact ingest, byte-identical fold across GPU splits, depth identical at
99.999 %, end-to-end scores equal to the reference's); the record is in docs/FINDINGS.md.

## License

MIT.
