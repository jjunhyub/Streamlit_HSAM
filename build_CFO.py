
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image


DATASET_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1")
IGNORE_DIRS = {"__subcrops"}

COLORED_SUFFIX = ".instances.colored.png"
MASK_ORIGINAL_FULL_SUFFIX = ".mask.original.full.png"
OVERLAY_SUFFIX = ".overlay.png"

OVERLAY_ALPHA = 0.45
OVERLAY_COLOR = np.array([255.0, 0.0, 0.0], dtype=np.float32)

PALETTE = np.array([
    [255,  99,  71],
    [135, 206, 235],
    [ 60, 179, 113],
    [255, 215,   0],
    [186,  85, 211],
    [255, 140,   0],
    [ 64, 224, 208],
    [220,  20,  60],
    [154, 205,  50],
    [ 70, 130, 180],
    [255, 105, 180],
    [205, 133,  63],
    [  0, 191, 255],
    [255,  69,   0],
    [ 50, 205,  50],
    [123, 104, 238],
    [255, 182, 193],
    [218, 165,  32],
], dtype=np.uint8)

INSTANCE_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)\.mask\.png$", re.IGNORECASE)


def human_label(node_id: str) -> str:
    return node_id.split("__")[-1] if "__" in node_id else node_id


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def load_mask01(path: Path, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as img:
        mask = img.convert("L")
        if target_hw is not None and mask.size != (target_hw[1], target_hw[0]):
            mask = mask.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.NEAREST)
        arr = np.asarray(mask)
    return arr > 0


def color_for_index(i: int) -> np.ndarray:
    base = PALETTE[i % len(PALETTE)].astype(np.int16)
    cycle = i // len(PALETTE)
    if cycle == 0:
        return base.astype(np.uint8)
    delta = ((cycle % 5) - 2) * 18
    out = np.clip(base + delta, 40, 255)
    return out.astype(np.uint8)


def find_full_image_path(image_dir: Path) -> Optional[Path]:
    preferred = image_dir / "image.root.jpg"
    if preferred.exists():
        return preferred

    candidates: List[Tuple[int, str, Path]] = []
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            continue
        name = p.name.lower()
        score = 0
        if name == "image.root.jpg":
            score += 100
        if any(k in name for k in ["original", "orig", "rgb", "scene", "full", "image", "img"]):
            score += 10
        if any(k in name for k in ["overlay", "mask", "crop", "subcrop"]):
            score -= 20
        candidates.append((score, name, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def collect_instance_masks(node_dir: Path, leaf: str) -> List[Path]:
    indexed: List[Tuple[int, Path]] = []
    for p in sorted(node_dir.glob(f"{leaf}_*.mask.png")):
        m = INSTANCE_RE.match(p.name)
        if not m:
            continue
        indexed.append((int(m.group("idx")), p))
    if indexed:
        return [p for _, p in sorted(indexed, key=lambda x: x[0])]

    merged = node_dir / f"{leaf}.mask.png"
    if merged.exists():
        return [merged]
    return []


def build_colored(instance_paths: List[Path], target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    masks: List[Tuple[int, Path, np.ndarray]] = []
    for p in instance_paths:
        mask01 = load_mask01(p, target_hw=target_hw)
        area = int(mask01.sum())
        if area <= 0:
            continue
        masks.append((area, p, mask01))

    if not masks:
        return None

    h, w = target_hw
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 큰 마스크 먼저, 작은 마스크 나중
    masks.sort(key=lambda x: (-x[0], x[1].name))
    for color_idx, (_, _, mask01) in enumerate(masks):
        rgb[mask01] = color_for_index(color_idx)
    return rgb


def build_mask_original_full(full_rgb: np.ndarray, full_mask01: np.ndarray) -> np.ndarray:
    out = np.zeros_like(full_rgb, dtype=np.uint8)
    out[full_mask01] = full_rgb[full_mask01]
    return out


def build_overlay(full_rgb: np.ndarray, full_mask01: np.ndarray) -> np.ndarray:
    base = full_rgb.astype(np.float32).copy()
    if np.any(full_mask01):
        base[full_mask01] = (
            (1.0 - OVERLAY_ALPHA) * base[full_mask01]
            + OVERLAY_ALPHA * OVERLAY_COLOR
        )
    return np.clip(base, 0, 255).astype(np.uint8)


def iter_image_dirs(dataset_root: Path) -> Iterator[Path]:
    for image_dir in sorted(dataset_root.iterdir()):
        if image_dir.is_dir():
            yield image_dir


def iter_node_dirs(image_dir: Path) -> Iterator[Path]:
    for node_dir in sorted(image_dir.iterdir()):
        if not node_dir.is_dir():
            continue
        if node_dir.name in IGNORE_DIRS or node_dir.name.startswith("__"):
            continue
        yield node_dir


def process_node(image_dir: Path, node_dir: Path) -> str:
    leaf = human_label(node_dir.name)

    full_image_path = find_full_image_path(image_dir)
    if full_image_path is None:
        return "skip:no-full-image"

    mask_path = node_dir / f"{leaf}.mask.png"
    if not mask_path.exists():
        return "skip:no-mask"

    full_rgb = load_rgb(full_image_path)
    h, w = full_rgb.shape[:2]
    full_mask01 = load_mask01(mask_path, target_hw=(h, w))
    if int(full_mask01.sum()) == 0:
        return "skip:empty-mask"

    instance_paths = collect_instance_masks(node_dir, leaf)
    colored = build_colored(instance_paths, target_hw=(h, w))
    if colored is None:
        return "skip:no-instance-mask"

    mask_original_full = build_mask_original_full(full_rgb, full_mask01)
    overlay = build_overlay(full_rgb, full_mask01)

    save_rgb(node_dir / f"{leaf}{COLORED_SUFFIX}", colored)
    save_rgb(node_dir / f"{leaf}{MASK_ORIGINAL_FULL_SUFFIX}", mask_original_full)
    save_rgb(node_dir / f"{leaf}{OVERLAY_SUFFIX}", overlay)

    return f"ok:{len(instance_paths)}"


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    total_images = 0
    total_nodes = 0
    ok = 0
    skipped = 0
    failed = 0
    skip_reasons: Dict[str, int] = {}

    for image_dir in iter_image_dirs(DATASET_ROOT):
        total_images += 1
        print(f"[IMAGE {total_images}] {image_dir.name}")

        for node_dir in iter_node_dirs(image_dir):
            total_nodes += 1
            try:
                status = process_node(image_dir, node_dir)
                if status.startswith("ok:"):
                    ok += 1
                else:
                    skipped += 1
                    skip_reasons[status] = skip_reasons.get(status, 0) + 1
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {node_dir.name}: {e}")

        print(f"  progress nodes={total_nodes} ok={ok} skipped={skipped} failed={failed}")

    print("\nDone.")
    print(f"images={total_images}, nodes={total_nodes}, ok={ok}, skipped={skipped}, failed={failed}")
    if skip_reasons:
        print("\nSkip reasons:")
        for k in sorted(skip_reasons):
            print(f"  {k}: {skip_reasons[k]}")


if __name__ == "__main__":
    main()
