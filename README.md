# Stacker

A CUDA multi-GPU focus stacker with a geometry report and optical-train
diagnostics for AutoFOV. Design: `docs/DESIGN.md`. Prototype results:
`docs/FINDINGS.md`.

## Phase A prototype (`proto/stackproto`, Python, runs on the Mac)

Setup: `python3 -m venv --system-site-packages .venv && .venv/bin/pip install pytest`
(needs numpy, scipy, opencv, matplotlib).

```
cd proto
# synthetic stack with known faults (presets: ideal, telecentric, faulty, macro)
../.venv/bin/python -m stackproto synth ../data/synth/test --preset faulty --set relief="'flat'"
# score the analysis against the truth
../.venv/bin/python -m stackproto validate ../data/synth/test --target flat --axis-px=-32.6,0
# real stack: geometry report + diagnostics (+ history CSV row)
../.venv/bin/python -m stackproto analyze /path/to/stack --step-um 1.25 --fov-mm 2.4 \
    --target flat --axis-pct=-3.75,0 --tag objective=10x --tag lock=locked
```

Outputs go to `<stack>_stacker/`: `report.json` (geometry for all three crop
policies, per-frame transforms, diagnostics) and `diagnostics.png`. One row is
appended to `history/diagnostics.csv`.

Tests: `.venv/bin/python -m pytest -q tests` (~30 s).

## Moving to another machine
See `docs/HANDOFF.md` (status, open problems, Windows/CUDA setup) and `CLAUDE.md`.
Comparison tools against Helicon's output live in `tools/bench/` (paths from env vars).
