from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

from supabase import create_client
from reviewer_config import REVIEW_USERS


# 직접 수정할 값들
SUPABASE_URL = "https://vutkowvxahuggkmhrtka.supabase.co"
SUPABASE_KEY = ""
SUPABASE_SECRET_KEY = ""

APP_MODULE = "streamlit_tree_reviewer_app"

# 로컬 dataset root
LOCAL_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1").resolve()

# Storage 안의 prefix
REMOTE_PREFIX = "coco_pilot_v1"

IMAGE_MANIFEST_TABLE = "image_manifest"
REVIEW_ASSIGNMENTS_TABLE = "review_assignments"

BATCH_SIZE = 100

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
appmod = importlib.import_module(APP_MODULE)


def chunked(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i+size] for i in range(0, len(items), size)]


def remoteize(local_path: Path) -> str:
    rel = local_path.relative_to(LOCAL_ROOT).as_posix()
    return f"{REMOTE_PREFIX}/{rel}"

def clear_assignments_for_reviewers(reviewer_ids: List[str]) -> None:
    for reviewer_id in reviewer_ids:
        supabase.table(REVIEW_ASSIGNMENTS_TABLE).delete().eq(
            "reviewer_id", reviewer_id
        ).execute()

def build_manifest_row(record: Dict[str, Any]) -> Dict[str, Any]:
    image_id = record["image_id"]
    image_dir = Path(record["image_dir"])

    full_path = appmod.get_full_image_path(record)
    full_size = appmod.read_image_size(full_path) if full_path and Path(full_path).exists() else None

    overlay_path = image_dir / "final_union_on_image.png"
    overlay_remote = remoteize(overlay_path) if overlay_path.exists() else None

    nodes_out: Dict[str, Any] = {}

    for node_id, info in record["nodes"].items():
        bbox = appmod.find_node_bbox(record, node_id)

        node_payload = {
            "id": info["id"],
            "label": info["label"],
            "parent": info["parent"],
            "children": info["children"],
            "actual": info["actual"],
            "folder_name": info["folder_name"],
            "bbox": list(bbox) if bbox else None,
            "mask_path": None,
            "mask_original_path": None,
            "instance_paths": [],
            "instances_colored_path": None,
        }

        # 실제 노드만 asset 연결
        if info["actual"]:
            assets = local_node_assets(record, node_id)

            if assets.get("mask") and Path(assets["mask"]).exists():
                node_payload["mask_path"] = remoteize(Path(assets["mask"]))

            if assets.get("mask_original") and Path(assets["mask_original"]).exists():
                node_payload["mask_original_path"] = remoteize(Path(assets["mask_original"]))

            if assets.get("instances"):
                node_payload["instance_paths"] = [
                    remoteize(Path(p)) for p in assets["instances"] if Path(p).exists()
                ]
            if assets.get("instances_colored") and Path(assets["instances_colored"]).exists():
                node_payload["instances_colored_path"] = remoteize(Path(assets["instances_colored"]))

        nodes_out[node_id] = node_payload

    return {
        "image_id": image_id,
        "dataset_prefix": f"{REMOTE_PREFIX}/{image_id}",
        "root_image_path": remoteize(Path(full_path)) if full_path and Path(full_path).exists() else None,
        "root_overlay_path": overlay_remote,
        "full_size": list(full_size) if full_size else None,
        "roots": record["roots"],
        "actual_nodes": record["actual_nodes"],
        "nodes": nodes_out,
    }


def upsert_manifest(rows: List[Dict[str, Any]]) -> None:
    for batch in chunked(rows, BATCH_SIZE):
        supabase.table(IMAGE_MANIFEST_TABLE).upsert(
            batch,
            on_conflict="image_id",
        ).execute()


def build_assignment_rows(sorted_image_ids: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for reviewer_id, cfg in REVIEW_USERS.items():
        order = 0
        for start, end in cfg["ranges"]:
            for image_id in sorted_image_ids[start:end]:
                rows.append({
                    "reviewer_id": reviewer_id,
                    "image_id": image_id,
                    "sort_index": order,
                })
                order += 1

    return rows


def upsert_assignments(rows: List[Dict[str, Any]]) -> None:
    for batch in chunked(rows, BATCH_SIZE):
        supabase.table(REVIEW_ASSIGNMENTS_TABLE).upsert(
            batch,
            on_conflict="reviewer_id,image_id",
        ).execute()

def local_node_assets(record: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    node = record["nodes"].get(node_id, {})
    folder_name = node.get("folder_name")
    image_dir = Path(record["image_dir"])

    root_original = appmod.get_full_image_path(record)
    root_overlay = image_dir / "final_union_on_image.png"
    if not root_overlay.exists():
        root_overlay = None

    if not folder_name:
        return {
            "root_original": root_original,
            "root_overlay": root_overlay,
            "mask": None,
            "mask_original": None,
            "instances": [],
            "instances_colored": None,
        }

    node_dir = image_dir / folder_name
    leaf = appmod.human_label(node_id)

    mask_path = node_dir / f"{leaf}.mask.png"
    if not mask_path.exists():
        mask_path = None

    mask_original_candidates = [
        node_dir / f"{leaf}.bbox.jpg",
        node_dir / f"{leaf}.rgb.jpg",
        node_dir / f"{leaf}.crop.jpg",
        node_dir / f"{leaf}.jpg",
        node_dir / "bbox.jpg",
        node_dir / "crop.jpg",
        node_dir / "image.jpg",
    ]
    mask_original_path = next((p for p in mask_original_candidates if p.exists()), None)

    instances = []
    if node_dir.exists():
        for p in sorted(node_dir.iterdir()):
            if p.is_file() and appmod.is_instance_mask_file(p, leaf):
                instances.append(p)

    instances_colored_candidates = [
        node_dir / f"{leaf}.instances.colored.png",
        node_dir / f"{leaf}.instances.colored.jpg",
        node_dir / f"{leaf}.instances.colored.webp",
    ]
    instances_colored_path = next(
        (p for p in instances_colored_candidates if p.exists()),
        None,
    )

    return {
        "root_original": root_original,
        "root_overlay": root_overlay,
        "mask": mask_path,
        "mask_original": mask_original_path,
        "instances": instances,
        "instances_colored": instances_colored_path,
    }


def main():
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"LOCAL_ROOT not found: {LOCAL_ROOT}")

    print(f"Scanning local dataset: {LOCAL_ROOT}")
    records = appmod.scan_dataset(LOCAL_ROOT)

    if not records:
        raise RuntimeError("No records found. Check LOCAL_ROOT.")

    manifest_rows = []
    sorted_image_ids = sorted(records.keys())

    for idx, image_id in enumerate(sorted_image_ids, start=1):
        manifest_rows.append(build_manifest_row(records[image_id]))
        if idx % 50 == 0 or idx == len(sorted_image_ids):
            print(f"[manifest] prepared {idx}/{len(sorted_image_ids)}")

    print("Uploading image_manifest rows...")
    upsert_manifest(manifest_rows)

    assignment_rows = build_assignment_rows(sorted_image_ids)

    reviewer_ids = list(REVIEW_USERS.keys())
    print(f"Clearing old review_assignments for reviewers: {reviewer_ids}")
    clear_assignments_for_reviewers(reviewer_ids)

    print(f"Uploading review_assignments rows... total={len(assignment_rows)}")
    upsert_assignments(assignment_rows)

    preview_path = Path("manifest_preview.json")
    preview_path.write_text(
        json.dumps(manifest_rows[:3], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"image_manifest rows: {len(manifest_rows)}")
    print(f"review_assignments rows: {len(assignment_rows)}")
    print(f"Preview saved: {preview_path.resolve()}")


if __name__ == "__main__":
    main()