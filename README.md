# xLamina

![xLamina](gui/icon/xlamina-128.png)

A photo stacker that can utilize multiple GPUs for rapidly stacking uncompressed 16-bit TIFF photos. It is a Windows program (C++ and Qt) built on CUDA for NVIDIA GPUs.

Open a folder of TIFF frames, choose the output algorithms, press Render, watch the stacks build in parallel for each method, and save. The program performs best when the frames sit on a RAM disk and the machine has enough RAM to also hold them resident (the frames are read once and kept pinned in memory): about twice the stack's size, 60 GB for 163 full-frame files of 180 MB. 

Methods: B (a regularised depth map: focus volume, semi-global matching, fine refinement,
soft-label composition), C (Laplacian pyramid), Cn (noise-aware pyramid), A (weighted
average) and several variants combining these in different ways; B0, CB, CBn, Cd, Cg, Cg2, H. Every render also writes a
geometry report: per-frame similarity transforms, the reference frame, the cropped values and the
effective µm/px over the stack depth when FOV and step size are known. 

## Install

Download `xLamina-<version>-win64.zip` from the Releases page, unzip it anywhere and run
`xLamina.exe`. Requirements: 64-bit Windows, an NVIDIA GPU (Ampere or newer) with a
current driver. 

## Use

Open a folder (or drop one onto the window). Every TIFF of the stack's image size is a
frame; other TIFFs in the folder are listed greyed out and left alone. Tick the outputs
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

After a render each frame in the list shows its wobble: the shift (px) and scale change
(%) from the previous frame, from the alignment. Frames whose shift is an outlier are
shown in red; Drop flagged unticks them, and unticked frames are left out of the next
render.

Smoothing (C) sets how much of C's weak detail is taken from B rather than from the
sharpest frame: 0 is the raw selection (in flat areas it picks the noisiest frame),
2 is about Helicon's Smoothing 1, 3 is Cn.

Add comparison image… (File menu, or drop a file onto a finished render) loads another
stacker's output as a tab beside xLamina's, so the two flip at the same zoom and position;
PgUp / PgDown step through the tabs.

Advanced (depth and composition), unticked by default, exposes the parameters of B's depth
pipeline and of the composition (depth resolution, focus window and power, depth window,
volume aggregation, refinement range and window, label softness, halo and dark priors, the
dark prior's reach, flat smoothing) at their defaults, for one-at-a-time experiments against
a 1:1 comparison. The depth volume is estimated at 1/2 of the sensor resolution since 0.6.1
(1/4 left doubled outlines on bright plates); it needs about 5 GB of GPU memory for 163
full-size frames. Since 0.6.3 the dark prior stays off within 24 px of sharp content (on the
finer grid it had filled the gaps of needle fans with far frames). Since 0.6.4 defocused
background is handled like the reference stacker's: beside sharp detail it keeps that detail's
frame for about 128 px (flat smoothing), beyond that (since 0.6.16) each pixel takes the frame
where its own focus measure peaks, smoothed blob by blob as the reference stacker does, with
soft transitions instead of patches of different frames with visible seams; since 0.6.18 the
composition blends up to 16 frames there, so a depth ramp draws no rings. The warp's
resampling filter is a choice since 0.6.14 (bicubic, the reference; Lanczos 3 or 8 for a little
more fine detail; bilinear). The 0.5.x B is `--b-reference 1` on the command line (depth
resolution 1/4, dark prior everywhere, none of the 0.6.x additions).

Cn is C at smoothing 3; with the panel's C smoothing at 2 the two differ on about 3 % of
pixels, so tick Cn only when C's smoothing is lower.

Performance holds the hardware settings: which GPUs, streams per GPU, reader threads,
alignment pairs in flight, and whether the frames stay resident. None of them changes the
result (B and C are bit-identical across all of them). Tools > Benchmark (about 20 s)
times up to 40 frames of the open stack at half resolution under the current settings,
each GPU alone, and with the frames streamed instead of resident, and applies the fastest;
its thorough mode runs the whole stack at full resolution with every setting varied in
turn and then the best of each combined. A built-in synthetic stack stands in when no
folder is open, or to compare machines. The report can be saved or copied.

Live view costs a little speed (each image copied off a GPU costs its uploads about 3 ms,
so views are rate-limited); Off is the fastest setting.

The frames are read once and kept pinned in host memory for the rest of the job (and, in
the GUI, for the next render of the same folder), as long as they fit in about half the
machine's RAM; the GPUs then pull them at their PCIe rate instead of the disk's.

All settings, the window layout, the theme (View > Theme: dark, light or the system style)
and the recent folders are kept in `%APPDATA%\xLamina\xLamina.ini`.

The same engine runs from the command line:

    stackcli run <folder> [--out DIR] [--outputs B,C,Cn] [--fov-mm X] [--scale 0.5] [--interp bicubic] [--smoothing 2] [--cache 0] ...

## Build

Visual Studio 2022 (Build Tools with the C++ workload suffice), CUDA 13.x, and for the
GUI Qt 6.8 for MSVC 2022 (`pip install aqtinstall` then
`aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 -O C:\Qt`).

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
