"""Frame discovery and loading (TIFF 8/16-bit, JPEG, PNG) via OpenCV."""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frames(folder, pattern: str = "*") -> list[Path]:
    folder = Path(folder)
    files = [p for p in folder.glob(pattern)
             if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")
             and not p.stem.startswith("_")]          # _truth_*.tif etc. are not frames
    return sorted(files, key=lambda p: _natural_key(p.name))


def read_rgb(path) -> np.ndarray:
    """Float32 RGB in [0, 1], shape (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.dtype == np.uint16:
        f = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        f = img.astype(np.float32) / 255.0
    else:
        f = img.astype(np.float32)
    if f.ndim == 2:
        f = np.repeat(f[:, :, None], 3, axis=2)
    elif f.shape[2] == 4:
        f = f[:, :, :3]
    return np.ascontiguousarray(f[:, :, ::-1])          # BGR -> RGB


def write_rgb16(path, rgb: np.ndarray) -> None:
    u16 = (np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    if not cv2.imwrite(str(path), np.ascontiguousarray(u16[:, :, ::-1])):
        raise IOError(f"cannot write {path}")


def luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def _work_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(16, int(round(width * scale))), max(16, int(round(height * scale)))


def load_luma_stack(paths, scale: float, cache_dir=None, workers: int | None = None):
    """Luma of every frame, downsampled by ``scale`` (pixel-area convention).

    Returns ``(stack, (W_full, H_full), (fx, fy))`` where ``stack`` is an
    (N, h, w) float32 array (memory-mapped from the cache when one is given)
    and ``fx = w/W``, ``fy = h/H`` are the exact factors centred coordinates
    scale by per axis (they differ slightly when w, h are rounded).
    """
    paths = [Path(p) for p in paths]
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    H, W = first.shape[:2]
    w, h = _work_size(W, H, scale)
    exact = (w / W, h / H)

    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig = hashlib.sha1(json.dumps(
            [[str(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in paths]
            + [w, h]).encode()).hexdigest()[:16]
        cache = cache_dir / f"luma_{w}x{h}_{sig}.npy"
        if cache.exists():
            arr = np.load(cache, mmap_mode="r")
            if arr.shape == (len(paths), h, w):
                return arr, (W, H), exact

    def one(p):
        img = read_rgb(p)
        if img.shape[:2] != (H, W):
            raise ValueError(f"{p.name}: size {img.shape[1]}x{img.shape[0]} != {W}x{H}")
        return cv2.resize(luma(img), (w, h), interpolation=cv2.INTER_AREA)

    out = np.empty((len(paths), h, w), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 4)) as ex:
        for k, im in enumerate(ex.map(one, paths)):
            out[k] = im
    if cache is not None:
        np.save(cache, out)
        out = np.load(cache, mmap_mode="r")
    return out, (W, H), exact
