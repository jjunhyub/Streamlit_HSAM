from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional


IGNORE_DIRS = {"__subcrops"}
CHUNK_SIZE = 100


def human_label(node_id: str) -> str:
    return node_id.split("__")[-1] if "__" in node_id else node_id


def collect_image_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def collect_node_folder_names(image_dir: Path) -> List[str]:
    return sorted(
        [
            p.name
            for p in image_dir.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and p.name not in IGNORE_DIRS
        ]
    )


def load_tree_text_if_exists(image_dir: Path) -> Optional[str]:
    tree_txt = image_dir / "tree.txt"
    if not tree_txt.exists():
        return None
    try:
        return tree_txt.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def build_tree_from_node_folders(node_folder_names: List[str]) -> Dict[str, List[str]]:
    children_map: Dict[str, List[str]] = {}

    def ensure_node(node_id: str) -> None:
        if node_id not in children_map:
            children_map[node_id] = []

    for folder in sorted(set(node_folder_names)):
        parts = folder.split("__")
        for idx in range(1, len(parts) + 1):
            node_id = "__".join(parts[:idx])
            ensure_node(node_id)

            if idx > 1:
                parent_id = "__".join(parts[: idx - 1])
                ensure_node(parent_id)
                if node_id not in children_map[parent_id]:
                    children_map[parent_id].append(node_id)

    for node_id in children_map:
        children_map[node_id] = sorted(children_map[node_id])

    return children_map


def render_tree_text(children_map: Dict[str, List[str]]) -> str:
    lines: List[str] = ["└─ root"]

    def rec(node_id: str, prefix: str, is_last: bool) -> None:
        label = human_label(node_id)
        branch = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{branch}{label}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        children = children_map.get(node_id, [])
        for idx, child_id in enumerate(children):
            rec(child_id, child_prefix, idx == len(children) - 1)

    root_children = children_map.get("root", [])
    for idx, child_id in enumerate(root_children):
        rec(child_id, "   ", idx == len(root_children) - 1)

    return "\n".join(lines)


def build_image_payload(image_dir: Path) -> Optional[Dict]:
    image_id = image_dir.name

    tree_text = load_tree_text_if_exists(image_dir)
    if tree_text:
        return {
            "image_id": image_id,
            "tree": tree_text,
        }

    node_folder_names = collect_node_folder_names(image_dir)
    if not node_folder_names:
        return None

    children_map = build_tree_from_node_folders(node_folder_names)
    tree_text = render_tree_text(children_map)

    return {
        "image_id": image_id,
        "tree": tree_text,
    }


def chunk_list(items: List[Dict], chunk_size: int) -> List[List[Dict]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def save_chunked_bundles(images: List[Dict], out_dir: Path, chunk_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_list(images, chunk_size)
    total_parts = len(chunks)
    pad = max(3, len(str(total_parts)))

    for part_idx, chunk in enumerate(chunks, start=1):
        bundle = {
            "schema_version": "hsam_tree_bundle_v1",
            "chunk_size": chunk_size,
            "part_index": part_idx,
            "part_count": total_parts,
            "image_count": len(chunk),
            "images": chunk,
        }

        out_path = out_dir / f"request_trees_only_part{part_idx:0{pad}d}.json"
        out_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        first_id = chunk[0]["image_id"] if chunk else "N/A"
        last_id = chunk[-1]["image_id"] if chunk else "N/A"
        print(
            f"[{part_idx}/{total_parts}] saved {len(chunk)} images "
            f"({first_id} ~ {last_id}) -> {out_path}"
        )


def main() -> None:
    dataset_root = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1")
    out_dir = Path(r"C:\Users\junhy\Documents\2026\HSAM\request_trees_only_chunks")

    images: List[Dict] = []
    for image_dir in collect_image_dirs(dataset_root):
        payload = build_image_payload(image_dir)
        if payload is not None:
            images.append(payload)

    save_chunked_bundles(images, out_dir, CHUNK_SIZE)

    print(f"done: total {len(images)} images, chunk_size={CHUNK_SIZE}")


if __name__ == "__main__":
    main()