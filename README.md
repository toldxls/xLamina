# xLamina

![xLamina](gui/icon/xlamina-128.png)

Multi-GPU focus stacker for photomicroscopy. Windows, NVIDIA GPUs, 16-bit TIFF in and out.

Open a folder of frames, choose the outputs, press Render, watch the stack build, save.
163 frames of 6960 x 4640 on an RTX 3080 Ti + RTX 3070: 10 s to align, 17 s to fuse.

Methods: B (regularised depth map: focus volume, semi-global matching, fine refinement,
soft-label composition), C (Laplacian pyramid), Cn (noise-aware pyramid), A (weighted
average) and the variants B0, CB, CBn, Cd, Cg, Cg2, H. Every render also writes a
geometry report: per-frame similarity transforms, the reference frame, the crop and the
effective µm/px over the stack depth.

## Install

Download `xLamina-<version>-win64.zip` from the Releases page, unzip it anywhere and run
`xLamina.exe`. Requirements: 64-bit Windows, an NVIDIA GPU (Ampere or newer) with a
current driver. Nothing else.

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

The frames are read once and kept pinned in host memory for the rest of the job (and, in
the GUI, for the next render of the same folder), as long as they fit in about half the
machine's RAM; the GPUs then pull them at their PCIe rate instead of the disk's.

The same engine runs from the command line:

    stackcli run <folder> [--out DIR] [--outputs B,C,Cn] [--fov-mm X] [--scale 0.5] [--smoothing 2] [--cache 0] ...

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
