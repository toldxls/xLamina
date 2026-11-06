"""Write the frame list stackcli fold wants (path zr zi tx ty per line) from an analyze report.

    python tools/validate/make_list.py tests/data/report_2026-11-05.json Q:\\TIFF list.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

rep = json.loads(Path(sys.argv[1]).read_text())
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for p in rep["per_frame"]:
        z = p["s"] * np.exp(1j * np.deg2rad(p["theta_deg"]))
        f.write(f"{Path(sys.argv[2]) / p['file']} {z.real:.17g} {z.imag:.17g} {p['tx']:.17g} {p['ty']:.17g}\n")
print(f"{len(rep['per_frame'])} frames -> {sys.argv[3]}")
