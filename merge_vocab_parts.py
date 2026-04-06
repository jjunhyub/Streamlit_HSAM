from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


PART_PATTERN = re.compile(
    r"request_trees_only_part(\d{3})_translated_reviewed\.json$"
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_part_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and PART_PATTERN.match(p.name):
            files.append(p)

    files.sort(key=lambda p: int(PART_PATTERN.match(p.name).group(1)))  # type: ignore[union-attr]
    return files


def validate_part_structure(path: Path, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: top-level must be a dict")

    if "images" not in data:
        raise ValueError(f"{path.name}: missing 'images' key")

    images = data["images"]
    if not isinstance(images, list):
        raise ValueError(f"{path.name}: 'images' must be a list")

    for idx, item in enumerate(images):
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}: images[{idx}] must be a dict")
        if "image_id" not in item:
            raise ValueError(f"{path.name}: images[{idx}] missing 'image_id'")
        if "nodes" not in item:
            raise ValueError(f"{path.name}: images[{idx}] missing 'nodes'")

    return images


def merge_parts(folder: Path, out_name: str = "request_trees_only_all_translated_reviewed.json") -> Path:
    part_files = find_part_files(folder)
    if not part_files:
        raise FileNotFoundError(
            f"No matching part files found in: {folder}"
        )

    merged_images: List[Dict[str, Any]] = []
    seen_image_ids: Dict[str, Path] = {}

    print(f"Found {len(part_files)} part files:")
    for p in part_files:
        print(f"  - {p.name}")

    for part_path in part_files:
        data = load_json(part_path)
        images = validate_part_structure(part_path, data)

        for item in images:
            image_id = str(item["image_id"])

            if image_id in seen_image_ids:
                prev = seen_image_ids[image_id]
                raise ValueError(
                    f"Duplicate image_id '{image_id}' found in both "
                    f"{prev.name} and {part_path.name}"
                )

            seen_image_ids[image_id] = part_path
            merged_images.append(item)

    merged = {"images": merged_images}
    out_path = folder / out_name

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved merged file to: {out_path}")
    print(f"Total images: {len(merged_images)}")

    return out_path


if __name__ == "__main__":
    folder = Path("vocab_files_fixed")
    merge_parts(folder)