from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


# 직접 수정 없이 바로 실행하는 고정 경로
DATASET_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1")

# 생성 파일명 규칙
OUTPUT_SUFFIX = ".instances.colored.png"
IGNORE_DIRS = {"__subcrops"}

# 너무 어두운 색/검정 계열을 피한 팔레트
PALETTE = np.array([
    [255,  99,  71],   # tomato
    [135, 206, 235],   # skyblue
    [ 60, 179, 113],   # mediumseagreen
    [255, 215,   0],   # gold
    [186,  85, 211],   # mediumorchid
    [255, 140,   0],   # darkorange
    [ 64, 224, 208],   # turquoise
    [220,  20,  60],   # crimson
    [154, 205,  50],   # yellowgreen
    [ 70, 130, 180],   # steelblue
    [255, 105, 180],   # hotpink
    [205, 133,  63],   # peru
    [  0, 191, 255],   # deepskyblue
    [255,  69,   0],   # orangered
    [ 50, 205,  50],   # limegreen
    [123, 104, 238],   # mediumslateblue
    [255, 182, 193],   # lightpink
    [218, 165,  32],   # goldenrod
], dtype=np.uint8)


INSTANCE_RE = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)\.mask\.png$", re.IGNORECASE)


def load_mask01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)



def color_for_index(i: int) -> np.ndarray:
    base = PALETTE[i % len(PALETTE)].astype(np.int16)
    cycle = i // len(PALETTE)
    if cycle == 0:
        return base.astype(np.uint8)

    # 팔레트 반복 시 밝기만 조금씩 흔들어서 동일색 반복 완화
    delta = ((cycle % 5) - 2) * 18
    out = np.clip(base + delta, 40, 255)
    return out.astype(np.uint8)



def find_base_label(node_dir: Path) -> str:
    return node_dir.name.split("__")[-1]



def collect_instance_masks(node_dir: Path) -> Tuple[str, List[Path]]:
    """
    우선순위:
    1) label_1.mask.png, label_2.mask.png ... 같은 개별 인스턴스들
    2) 없으면 label.mask.png 하나를 단일 인스턴스로 사용
    """
    base_label = find_base_label(node_dir)

    instance_paths: List[Tuple[int, Path]] = []
    for p in sorted(node_dir.glob(f"{base_label}_*.mask.png")):
        m = INSTANCE_RE.match(p.name)
        if not m:
            continue
        instance_paths.append((int(m.group("idx")), p))

    if instance_paths:
        return base_label, [p for _, p in sorted(instance_paths, key=lambda x: x[0])]

    merged = node_dir / f"{base_label}.mask.png"
    if merged.exists():
        return base_label, [merged]

    return base_label, []



def composite_instances(instance_paths: List[Path]) -> np.ndarray | None:
    masks: List[Tuple[int, Path, np.ndarray]] = []
    shape = None

    for p in instance_paths:
        mask01 = load_mask01(p)
        area = int(mask01.sum())
        if area <= 0:
            continue
        if shape is None:
            shape = mask01.shape
        elif shape != mask01.shape:
            raise ValueError(f"Mask shape mismatch: {p}")
        masks.append((area, p, mask01))

    if not masks:
        return None

    h, w = shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 중요: 큰 인스턴스를 먼저 깔고, 작은 인스턴스를 나중에 덮어쓴다.
    # 포함관계가 있을 때 작은 내부 인스턴스가 살아남는다.
    masks.sort(key=lambda x: (-x[0], x[1].name))

    for color_idx, (_, _, mask01) in enumerate(masks):
        rgb[mask01] = color_for_index(color_idx)

    return rgb



def process_node_dir(node_dir: Path) -> str:
    base_label, instance_paths = collect_instance_masks(node_dir)
    if not instance_paths:
        return "skip:no-mask"

    out_path = node_dir / f"{base_label}{OUTPUT_SUFFIX}"
    rgb = composite_instances(instance_paths)
    if rgb is None:
        return "skip:empty-mask"

    save_rgb(out_path, rgb)
    return f"ok:{len(instance_paths)}"



def iter_node_dirs(dataset_root: Path):
    for image_dir in sorted(dataset_root.iterdir()):
        if not image_dir.is_dir():
            continue
        for node_dir in sorted(image_dir.iterdir()):
            if not node_dir.is_dir():
                continue
            if node_dir.name in IGNORE_DIRS:
                continue
            if node_dir.name.startswith("__"):
                continue
            yield node_dir



def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for node_dir in iter_node_dirs(DATASET_ROOT):
        total += 1
        try:
            status = process_node_dir(node_dir)
            if status.startswith("ok:"):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {node_dir}: {e}")

        if total % 200 == 0:
            print(f"[{total}] ok={ok}, skipped={skipped}, failed={failed}")

    print("\nDone.")
    print(f"total={total}, ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
