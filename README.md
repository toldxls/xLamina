# xLamina

Multi-GPU focus stacker for photomicroscopy. CUDA engine, Windows, 16-bit TIFF in and out.

Methods: A (weighted average), B (regularised depth map: focus volume, semi-global
matching, fine refinement, soft-label composition), C (Laplacian pyramid) and the
variants CB, Cn, CBn, Cd, Cg, Cg2, H. Every run also writes a geometry report:
per-frame similarity transforms, the reference frame, the crop and the effective
µm/px over the stack depth.

163 frames of 6960 x 4640 on an RTX 3080 Ti + RTX 3070: alignment 10 s, fusion
18 s for all outputs.

## Build

Visual Studio 2022 Build Tools (C++ workload), CUDA 13.x, CMake, Ninja.

    tools\build.cmd

produces `build\cli\stackcli.exe`. It depends only on the NVIDIA driver and the
VC++ runtime.

## Use

    stackcli run <folder> [--out DIR] [--outputs B,C,Cn] [--fov-mm X | --um-per-px X]
                 [--scale 1|0.5] [--ref middle] [--lock-scale 1] [--gpus 0,1]

Aligns the `IMG_*.tif` frames in the folder, picks the reference and the crop, and
writes `report.json` and the outputs as 16-bit TIFFs to `<folder>_stacker`.
`stackcli` without arguments lists the developer commands (one per pipeline stage).

## Layout

    core/     stackcore, the C++/CUDA engine
    cli/      stackcli
    proto/    the Python reference implementation (numpy, scipy, OpenCV)
    tools/    benchmarks (m0), the comparison harness, validation scripts
    docs/     DESIGN.md, FINDINGS.md (evidence behind every decision), HANDOFF.md

The CUDA port reproduces the prototype stage by stage; `tools/validate` holds the
checks (bit-exact ingest, 1e-6 warp, byte-identical fold across GPU splits, depth
identical at 99.999 %, end-to-end scores equal to the prototype's).

## License

MIT.
