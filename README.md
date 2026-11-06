# xLamina

Multi-GPU focus stacker for photomicroscopy. CUDA engine, Windows, 16-bit TIFF in and out.
Command line for now; a GUI is planned.

Methods: A (weighted average), B (regularised depth map: focus volume, semi-global
matching, fine refinement, soft-label composition), C (Laplacian pyramid) and the
variants CB, Cn, CBn, Cd, Cg, Cg2, H. Every run also writes a geometry report:
per-frame similarity transforms, the reference frame, the crop and the effective
µm/px over the stack depth.

163 frames of 6960 x 4640 on an RTX 3080 Ti + RTX 3070: alignment 10 s, fusion
17 s for all outputs.

## Install

Download `xLamina-<version>-win64.zip` from the Releases page and unzip it anywhere.
It needs a 64-bit Windows with an NVIDIA GPU (Ampere or newer) and a current driver;
nothing else.

## Use

Open a terminal in the unzipped folder:

    stackcli run <folder> [--out DIR] [--outputs B,C,Cn] [--fov-mm X | --um-per-px X]
                 [--prefix IMG_] [--scale 1|0.5] [--ref middle] [--lock-scale 1] [--gpus 0,1]

Every `.tif` in the folder is a frame (or only those whose name starts with `--prefix`).
The frames are aligned, the reference and crop chosen, and `report.json` plus the
requested outputs (default: all) are written as 16-bit TIFFs to `<folder>_stacker`.
`--fov-mm` is the horizontal field of view of one frame; it gives the report its µm/px.
`stackcli` without arguments lists the developer commands, one per pipeline stage.

## Build

Visual Studio 2022 (Build Tools with the C++ workload suffice), CUDA 13.x.

    tools\build.cmd        builds build\cli\stackcli.exe
    tools\package.cmd      builds and zips a release into dist\

## Layout

    core/     stackcore, the C++/CUDA engine
    cli/      stackcli
    tools/    build and packaging scripts, the M0 hardware benchmark
    docs/     DESIGN.md and FINDINGS.md (the evidence behind every decision)

The engine was ported stage by stage from a Python reference implementation and
checked against it (bit-exact ingest, byte-identical fold across GPU splits, depth
identical at 99.999 %, end-to-end scores equal to the reference's); the record is in
docs/FINDINGS.md.

## License

MIT.
