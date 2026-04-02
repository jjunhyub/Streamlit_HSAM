from __future__ import annotations

# Standard library
import gzip
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

# Third-party
import numpy as np
import streamlit as st
from PIL import Image
from supabase import Client, create_client

# Local
from reviewer_config import REVIEW_USERS


# Internal config
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
QUESTION_PAGE_SIZE = 8
APP_TITLE = "H-SAM Review Tool"
TREE_BUTTON_MIN_WIDTH_PX = 10
TREE_BUTTON_MAX_WIDTH_PX = 220
TREE_BUTTON_BASE_PX = 30
TREE_CHAR_WIDTH_PX = 6
TREE_BUTTON_HEIGHT_PX = 30
TREE_CONNECTOR_MARGIN_PX = 4

TREE_SIBLING_GAP_PX = 0 # 자식끼리의 간격
TREE_ROOT_GAP_PX = 1 # 루트끼리의 GAP
TREE_SIDE_PAD_PX = 10 # 옆으로 얼마나 갈건지

TREE_ROW_HEIGHT_PX = 50
TREE_ROW_GAP_PX = 30
TREE_PANEL_TOP_PAD_PX = 0

TREE_SUMMARY_NODE_ID = "__tree_summary__"

# Fixed path
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "datasets" / "coco_pilot_v1"
OUTPUT_ROOT = BASE_DIR / "outputs"
# DATASET_ROOT = Path("../datasets/coco_pilot_v1")
# OUTPUT_ROOT = Path("./outputs")

# Utilities

def safe_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)

def human_label(node_id: str) -> str:
    return node_id.split("__")[-1] if "__" in node_id else node_id

def now_iso():
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

# Tree helper


def get_node_depth(node_id: str) -> int:
    return max(0, node_id.count("__"))

def get_node_path_labels(node_id: str) -> List[str]:
    return node_id.split("__")

def get_parent_node_id(record: Dict[str, Any], node_id: str) -> Optional[str]:
    return record["nodes"].get(node_id, {}).get("parent")

def is_reviewable_node(node_id: str) -> bool:
    return human_label(node_id).lower() != "others"

def first_reviewable_node_id(record: Dict[str, Any]) -> Optional[str]:
    for node_id in record["actual_nodes"]:
        if is_reviewable_node(node_id):
            return node_id
    return None

def tree_display_children(record: Dict[str, Any], node_id: str) -> List[str]:
    out: List[str] = []
    for child_id in record["nodes"][node_id]["children"]:
        child_node = record["nodes"][child_id]
        if child_node["actual"] and not is_reviewable_node(child_id):
            out.extend(tree_display_children(record, child_id))
        else:
            out.append(child_id)
    return out


def display_root_ids(record: Dict[str, Any]) -> List[str]:
    roots = list(record["roots"])
    if not roots:
        return [TREE_SUMMARY_NODE_ID]
    return roots

# File / image helper

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json_maybe_gz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

def pil_to_displayable(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA", "L"):
        return img.convert("RGBA")
    return img

def score_scene_file(path: Path) -> Tuple[int, str]:
    name = path.name.lower()
    score = 0
    if any(k in name for k in ["original", "orig", "rgb", "scene", "full", "image", "img"]):
        score += 5
    if any(k in name for k in ["overlay", "mask", "b01", "crop", "subcrop"]):
        score -= 3
    return (-score, name)

def scene_images_for_record(record: Dict[str, Any]) -> List[Path]:
    base = Path(record["image_dir"])
    files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=score_scene_file)

def is_instance_mask_file(path: Path, leaf: str) -> bool:
    name = path.name.lower()
    leaf = leaf.lower()
    return bool(re.fullmatch(rf"{re.escape(leaf)}_\d+\.mask\.png", name))

def instance_mask_sort_key(path: Path, leaf: str) -> Tuple[int, str]:
    name = path.name.lower()
    m = re.fullmatch(rf"{re.escape(leaf.lower())}_(\d+)\.mask\.png", name)
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)

def get_full_image_path(record: Dict[str, Any]) -> Optional[Path]:
    base = Path(record["image_dir"])
    candidate = base / "image.root.jpg"
    if candidate.exists():
        return candidate

    files = scene_images_for_record(record)
    for p in files:
        if p.name.lower() == "image.root.jpg":
            return p
    return files[0] if files else None

def get_full_image_size(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    full_path = get_full_image_path(record)
    if not full_path:
        return None
    return read_image_size(full_path)

def extract_bbox_from_payload(payload: Any, leaf: str) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(payload, dict):
        # direct bbox keys
        if all(k in payload for k in ("x1", "y1", "x2", "y2")):
            try:
                return (int(payload["x1"]), int(payload["y1"]), int(payload["x2"]), int(payload["y2"]))
            except Exception:
                pass

        if "bbox" in payload:
            bbox = payload["bbox"]
            if isinstance(bbox, dict) and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                try:
                    return (int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
                except Exception:
                    pass
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    return tuple(int(v) for v in bbox)
                except Exception:
                    pass

        # leaf/name keyed
        for key in ("label", "name", "node_label", "leaf"):
            if str(payload.get(key, "")).lower() == leaf.lower():
                found = extract_bbox_from_payload({k: v for k, v in payload.items() if k != key}, leaf)
                if found:
                    return found

        for v in payload.values():
            found = extract_bbox_from_payload(v, leaf)
            if found:
                return found

    elif isinstance(payload, list):
        for item in payload:
            found = extract_bbox_from_payload(item, leaf)
            if found:
                return found

    return None

def find_node_bbox(record: Dict[str, Any], node_id: str) -> Optional[Tuple[int, int, int, int]]:
    node_info = record["nodes"].get(node_id, {})
    folder_name = node_info.get("folder_name")
    if not folder_name:
        return None

    leaf = human_label(node_id)
    node_dir = Path(record["image_dir"]) / folder_name

    # 1) node-local masks json
    local_masks_json = node_dir / f"{leaf}.masks.json"
    payload = load_json_maybe_gz(local_masks_json)
    bbox = extract_bbox_from_payload(payload, leaf)
    if bbox:
        return bbox

    # 2) node-local audit
    local_audit_gz = node_dir / "node.audit.json.gz"
    payload = load_json_maybe_gz(local_audit_gz)
    bbox = extract_bbox_from_payload(payload, leaf)
    if bbox:
        return bbox

    # 3) image-level tree / index / provenance fallback
    for fname in ["tree.json", "node_index.json", "provenance.json"]:
        payload = load_json_maybe_gz(Path(record["image_dir"]) / fname)
        bbox = extract_bbox_from_payload(payload, leaf)
        if bbox:
            return bbox

    return None

def restore_crop_to_full_canvas(
    crop_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    full_size: Tuple[int, int],
    fill: int = 0,
) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    full_w, full_h = [int(v) for v in full_size]

    crop = crop_img
    if crop.mode not in ("RGB", "RGBA", "L"):
        crop = crop.convert("RGBA")

    if crop.mode == "L":
        canvas = Image.new("L", (full_w, full_h), color=fill)
    elif crop.mode == "RGBA":
        canvas = Image.new("RGBA", (full_w, full_h), color=(0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (full_w, full_h), color=(fill, fill, fill))

    target_w = max(0, x2 - x1)
    target_h = max(0, y2 - y1)
    if target_w == 0 or target_h == 0:
        return canvas

    if crop.size != (target_w, target_h):
        crop = crop.resize((target_w, target_h))

    canvas.paste(crop, (x1, y1))
    return canvas

def collect_prefetch_paths_for_image(record: Dict[str, Any], include_instances: bool = False) -> List[str]:
    paths: List[str] = []

    def add_path(p: Optional[str]) -> None:
        if p and p not in paths:
            paths.append(p)

    add_path(record.get("root_image_path"))
    add_path(record.get("root_overlay_path"))

    for node_id in record.get("actual_nodes", []):
        node = record["nodes"].get(node_id, {})
        add_path(node.get("mask_path"))
        add_path(node.get("mask_original_path"))

        if include_instances:
            for p in node.get("instance_paths", []) or []:
                add_path(p)

    return paths

def prefetch_image_assets(record: Dict[str, Any], include_instances: bool = False, max_files: Optional[int] = None) -> None:
    image_id = record["image_id"]
    paths = collect_prefetch_paths_for_image(record, include_instances=include_instances)

    if max_files is not None:
        paths = paths[:max_files]

    for storage_path in paths:
        try:
            get_blob(image_id, storage_path)
        except Exception:
            pass

def build_mask_original_display(record: Dict[str, Any], node_id: str, storage_path: Optional[str]) -> Optional[Image.Image]:
    if not storage_path:
        return None

    assets = node_assets(record, node_id)
    full_size = assets.get("full_size")
    bbox = assets.get("bbox")

    try:
        crop = pil_from_storage_raw(record["image_id"], storage_path).convert("RGB")
        if not full_size or not bbox:
            return crop

        restored = restore_crop_to_full_canvas(
            crop_img=crop,
            bbox=bbox,
            full_size=full_size,
            fill=0,
        )
        return pil_to_displayable(restored)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_displayable_image_bytes(image_id: str, storage_path: str) -> bytes:
    raw = get_blob(image_id, storage_path)
    with Image.open(BytesIO(raw)) as img:
        if img.mode == "P":
            img = img.convert("RGBA")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        out = pil_to_displayable(img.copy())
        bio = BytesIO()
        if out.mode == "RGBA":
            out.save(bio, format="PNG")
        elif out.mode == "L":
            out.save(bio, format="PNG")
        else:
            out.save(bio, format="JPEG", quality=90)
        return bio.getvalue()

def build_direct_display(storage_path: Optional[str]) -> Optional[Image.Image]:
    image_id = st.session_state.get("selected_image_id")
    if not storage_path or not image_id:
        return None

    try:
        raw = get_displayable_image_bytes(image_id, storage_path)
        with Image.open(BytesIO(raw)) as img:
            return img.copy()
    except Exception:
        return None
    
# def build_direct_display(storage_path: Optional[str]) -> Optional[Image.Image]:
#     image_id = st.session_state.get("selected_image_id")
#     if not storage_path or not image_id:
#         return None

#     try:
#         img = pil_from_storage_raw(image_id, storage_path)
#         if img.mode == "P":
#             img = img.convert("RGBA")
#         elif img.mode not in ("RGB", "RGBA", "L"):
#             img = img.convert("RGB")
#         return pil_to_displayable(img)
#     except Exception:
#         return None

@st.cache_data(show_spinner=False)
def build_overlay_png_bytes(image_id: str, full_path: str, mask_path: str, alpha: float = 0.45) -> bytes:
    base = pil_from_storage_raw(image_id, full_path).convert("RGBA")
    mask = pil_from_storage_raw(image_id, mask_path).convert("L")

    base_np = np.array(base).copy()
    mask_np = np.array(mask)

    active = mask_np > 0
    overlay = base_np.copy()

    overlay_rgb = overlay[..., :3].astype(np.float32)
    color_rgb = np.zeros_like(overlay_rgb)
    color_rgb[..., 0] = 255.0

    overlay_rgb[active] = (
        (1.0 - alpha) * overlay_rgb[active] + alpha * color_rgb[active]
    )

    overlay[..., :3] = np.clip(overlay_rgb, 0, 255).astype(np.uint8)
    out = Image.fromarray(overlay, mode="RGBA")

    bio = BytesIO()
    out.save(bio, format="PNG")
    return bio.getvalue()

def build_overlay_on_full_image(
    record: Dict[str, Any],
    node_id: str,
    alpha: float = 0.45,
) -> Optional[Image.Image]:
    assets = node_assets(record, node_id)
    full_path = assets.get("root_original")
    mask_path = assets.get("mask")

    if not full_path or not mask_path:
        return None

    try:
        raw = build_overlay_png_bytes(record["image_id"], full_path, mask_path, alpha)
        with Image.open(BytesIO(raw)) as img:
            return img.copy()
    except Exception:
        return None
    
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def supabase_table_name() -> str:
    return st.secrets["supabase"].get("table", "review_annotations")

def dataset_bucket_name() -> str:
    return st.secrets["supabase"].get("bucket", "review-dataset")

def ensure_image_blob_cache(image_id: str) -> Dict[str, bytes]:
    if st.session_state.get("_blob_cache_image_id") != image_id:
        st.session_state["_blob_cache_image_id"] = image_id
        st.session_state["blob_cache"] = {}
    return st.session_state["blob_cache"]

def ensure_blob_cache() -> Dict[str, bytes]:
    if "blob_cache" not in st.session_state:
        st.session_state["blob_cache"] = {}
    return st.session_state["blob_cache"]

def download_blob(storage_path: str) -> bytes:
    return get_supabase_client().storage.from_(dataset_bucket_name()).download(storage_path)


# def get_blob(image_id: str, storage_path: str) -> bytes:
#     cache = ensure_image_blob_cache(image_id)
#     if storage_path not in cache:
#         cache[storage_path] = download_blob(storage_path)
#     return cache[storage_path]

def get_blob(image_id: str, storage_path: str) -> bytes:
    cache = ensure_blob_cache()
    cache_key = f"{image_id}::{storage_path}"
    if cache_key not in cache:
        cache[cache_key] = download_blob(storage_path)
    return cache[cache_key]

def pil_from_storage_raw(image_id: str, storage_path: str) -> Image.Image:
    raw = get_blob(image_id, storage_path)
    with Image.open(BytesIO(raw)) as img:
        return img.copy()

def load_annotations_from_supabase(reviewer_id: str) -> Dict[str, Any]:
    reviewer_id = safe_token(reviewer_id.strip() or "anonymous")
    sb = get_supabase_client()

    try:
        resp = (
            sb.table(supabase_table_name())
            .select("payload")
            .eq("reviewer_id", reviewer_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if rows:
            payload = rows[0].get("payload")
            if isinstance(payload, dict):
                return payload
    except Exception as e:
        st.session_state["last_db_save_ok"] = False
        st.session_state["last_db_error"] = f"load failed: {e}"

    return {}

def save_annotations_to_supabase(reviewer_id: str, payload: Dict[str, Any]) -> None:
    reviewer_id = safe_token(reviewer_id.strip() or "anonymous")
    sb = get_supabase_client()
    table = supabase_table_name()

    sb.table(table).upsert({
        "reviewer_id": reviewer_id,
        "payload": payload,
        "updated_at": now_iso()
    }).execute()

def maybe_autosave_to_supabase(force: bool = False) -> None:
    now_ts = time.time()
    last_ts = st.session_state.get("last_db_save_ts", 0.0)
    dirty = st.session_state.get("annotations_dirty", False)

    if not force:
        if not dirty:
            return
        if now_ts - last_ts < 15:
            return

    reviewer = st.session_state.reviewer_id
    save_annotations_to_supabase(reviewer, st.session_state.annotations)
    st.session_state["last_db_save_ts"] = now_ts
    st.session_state["annotations_dirty"] = False

def chunked(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]

def hierarchy_button_width_px(node_id: str) -> int:
    label = human_label(node_id)
    width = TREE_BUTTON_BASE_PX + TREE_CHAR_WIDTH_PX * len(label)
    return int(max(TREE_BUTTON_MIN_WIDTH_PX, min(TREE_BUTTON_MAX_WIDTH_PX, width)))



@st.cache_data(show_spinner=False)
def compute_hierarchy_layout_cached(record_sig: str) -> Dict[str, Any]:
    record = json.loads(record_sig)
    node_widths: Dict[str, float] = {}
    subtree_widths: Dict[str, float] = {}
    node_lefts: Dict[str, float] = {}
    rows_by_depth: Dict[int, List[str]] = {}

    def human_label_from_id(node_id: str) -> str:
        return node_id.split("__")[-1] if "__" in node_id else node_id

    def is_reviewable(node_id: str) -> bool:
        return human_label_from_id(node_id).lower() != "others"
    
    # def tree_children(node_id: str) -> List[str]:
    #     if node_id == TREE_SUMMARY_NODE_ID:
    #         return []

    #     out = []
    #     for child_id in record["nodes"][node_id]["children"]:
    #         child_node = record["nodes"][child_id]
    #         if child_node["actual"] and not is_reviewable(child_id):
    #             out.extend(tree_children(child_id))
    #         else:
    #             out.append(child_id)
    #     return

    def tree_children(node_id: str) -> List[str]:
        if node_id == TREE_SUMMARY_NODE_ID:
            return []

        out = []
        for child_id in record["nodes"][node_id]["children"]:
            child_node = record["nodes"][child_id]
            if child_node["actual"] and not is_reviewable(child_id):
                out.extend(tree_children(child_id))
            else:
                out.append(child_id)
        return out

    def button_width(node_id: str) -> int:
        if node_id == TREE_SUMMARY_NODE_ID:
            label = "전체평가"
        else:
            label = human_label_from_id(node_id)

        width = TREE_BUTTON_BASE_PX + TREE_CHAR_WIDTH_PX * len(label)
        return int(max(TREE_BUTTON_MIN_WIDTH_PX, min(TREE_BUTTON_MAX_WIDTH_PX, width)))

    def measure(node_id: str) -> float:
        own_width = button_width(node_id)
        node_widths[node_id] = own_width
        children = tree_children(node_id)
        if not children:
            subtree_width = own_width
        else:
            children_total = 0.0
            for idx, child_id in enumerate(children):
                children_total += measure(child_id)
                if idx < len(children) - 1:
                    children_total += TREE_SIBLING_GAP_PX
            subtree_width = max(own_width, children_total)
        subtree_widths[node_id] = subtree_width
        return subtree_width

    def place(node_id: str, left_x: float, depth: int) -> None:
        # node_lefts[node_id] = left_x
        node_lefts[node_id] = left_x + (subtree_widths[node_id] - node_widths[node_id]) / 2.0
        rows_by_depth.setdefault(depth, []).append(node_id)
        children = tree_children(node_id)
        if not children:
            return
        child_left = left_x
        for idx, child_id in enumerate(children):
            place(child_id, child_left, depth + 1)
            child_left += subtree_widths[child_id]
            if idx < len(children) - 1:
                child_left += TREE_SIBLING_GAP_PX

    # root_ids = record["roots"]
    root_ids = display_root_ids(record)
    if not root_ids:
        return {"rows": [], "node_lefts": {}, "node_widths": {}, "tree_width": 0.0}

    total_width = TREE_SIDE_PAD_PX * 2.0
    for idx, root_id in enumerate(root_ids):
        total_width += measure(root_id)
        if idx < len(root_ids) - 1:
            total_width += TREE_ROOT_GAP_PX

    left_x = TREE_SIDE_PAD_PX
    for idx, root_id in enumerate(root_ids):
        place(root_id, left_x, 0)
        left_x += subtree_widths[root_id]
        if idx < len(root_ids) - 1:
            left_x += TREE_ROOT_GAP_PX

    ordered_rows = []
    for depth in sorted(rows_by_depth):
        ordered_rows.append(sorted(rows_by_depth[depth], key=lambda nid: node_lefts[nid]))

    return {
        "rows": ordered_rows,
        "node_lefts": node_lefts,
        "node_widths": node_widths,
        "tree_width": total_width,
    }

@st.cache_data(show_spinner=False)
def build_tree_connector_svg_cached(
    image_id: str,
    rows: List[List[str]],
    node_lefts: Dict[str, float],
    node_widths: Dict[str, float],
    tree_width: float,
    node_children_map: Dict[str, List[str]],
    node_actual_map: Dict[str, bool],
) -> str:
    if not rows:
        return ""

    def is_reviewable(node_id: str) -> bool:
        return human_label(node_id).lower() != "others"

    def tree_children(node_id: str) -> List[str]:
        out = []
        for child_id in node_children_map.get(node_id, []):
            child_actual = node_actual_map.get(child_id, False)
            if child_actual and not is_reviewable(child_id):
                out.extend(tree_children(child_id))
            else:
                out.append(child_id)
        return out

    svg_height = (
        TREE_PANEL_TOP_PAD_PX * 2
        + len(rows) * TREE_ROW_HEIGHT_PX
        + max(0, len(rows) - 1) * TREE_ROW_GAP_PX
    )


    def button_top_y(row_idx: int) -> float:
        return row_top_y(row_idx) + (TREE_ROW_HEIGHT_PX - TREE_BUTTON_HEIGHT_PX) / 2.0

    def button_bottom_y(row_idx: int) -> float:
        return button_top_y(row_idx) + TREE_BUTTON_HEIGHT_PX
    
    def row_top_y(row_idx: int) -> float:
        return TREE_PANEL_TOP_PAD_PX + row_idx * (TREE_ROW_HEIGHT_PX + TREE_ROW_GAP_PX)

    def row_center_y(row_idx: int) -> float:
        return (
            TREE_PANEL_TOP_PAD_PX
            + row_idx * (TREE_ROW_HEIGHT_PX + TREE_ROW_GAP_PX)
            + TREE_ROW_HEIGHT_PX / 2.0
        )

    def node_center_x(node_id: str) -> float:
        return node_lefts[node_id] + node_widths[node_id] / 2.0

    strokes = []

    for row_idx, parent_row in enumerate(rows[:-1]):
        child_row = rows[row_idx + 1]
        parent_y = row_center_y(row_idx)
        child_y = row_center_y(row_idx + 1)
        parent_anchor_y = button_bottom_y(row_idx) + TREE_CONNECTOR_MARGIN_PX
        child_anchor_y = button_top_y(row_idx + 1) - TREE_CONNECTOR_MARGIN_PX
        bus_y = (parent_anchor_y + child_anchor_y) / 2.0
        # bus_y = child_y - TREE_ROW_HEIGHT_PX / 2.0 - 6.0
        child_set = set(child_row)

        for parent_id in parent_row:
            children = [c for c in tree_children(parent_id) if c in child_set]
            if not children:
                continue

            px = node_center_x(parent_id)

            if len(children) == 1:
                cx = node_center_x(children[0])
                strokes.append(
                    f"<line x1='{px:.1f}' y1='{parent_y + TREE_ROW_HEIGHT_PX/2 - 6:.1f}' "
                    f"x2='{cx:.1f}' y2='{child_y - TREE_ROW_HEIGHT_PX/2 + 6:.1f}' />"
                )
                continue

            child_centers = [node_center_x(c) for c in children]
            left_x = min(child_centers)
            right_x = max(child_centers)

            strokes.append(
                f"<line x1='{px:.1f}' y1='{parent_y + TREE_ROW_HEIGHT_PX/2 - 6:.1f}' "
                f"x2='{px:.1f}' y2='{bus_y:.1f}' />"
            )
            strokes.append(
                f"<line x1='{left_x:.1f}' y1='{bus_y:.1f}' "
                f"x2='{right_x:.1f}' y2='{bus_y:.1f}' />"
            )
            for cx in child_centers:
                strokes.append(
                    f"<line x1='{cx:.1f}' y1='{bus_y:.1f}' "
                    f"x2='{cx:.1f}' y2='{child_y - TREE_ROW_HEIGHT_PX/2 + 6:.1f}' />"
                )

    return f"""
    <div style="position:relative; height:0; overflow:visible; pointer-events:none;">
    <svg
        width="{int(math.ceil(tree_width))}"
        height="{int(math.ceil(svg_height))}"
        viewBox="0 0 {int(math.ceil(tree_width))} {int(math.ceil(svg_height))}"
        xmlns="http://www.w3.org/2000/svg"
        style="position:absolute; left:0; top:0; z-index:0; pointer-events:none; overflow:visible;">
        <g stroke="rgba(148,163,184,0.55)" stroke-width="2" fill="none" stroke-linecap="round">
            {''.join(strokes)}
        </g>
    </svg>
    </div>
    """

def inject_tree_summary_node(record: Dict[str, Any]) -> Dict[str, Any]:
    record = json.loads(json.dumps(record))  # deepcopy

    if TREE_SUMMARY_NODE_ID not in record["nodes"]:
        record["nodes"][TREE_SUMMARY_NODE_ID] = {
            "id": TREE_SUMMARY_NODE_ID,
            "label": "전체트리",
            "parent": None,
            "children": [],
            "actual": True,
        }

    if TREE_SUMMARY_NODE_ID not in record["roots"]:
        record["roots"] = [TREE_SUMMARY_NODE_ID] + record["roots"]

    return record

def record_structure_signature(record: Dict[str, Any]) -> str:
    payload = {
        "image_id": record["image_id"],
        "roots": record["roots"],
        "actual_nodes": record["actual_nodes"],
        "nodes": {
            nid: {
                "children": record["nodes"][nid].get("children", []),
                "actual": record["nodes"][nid].get("actual", False),
                "label": record["nodes"][nid].get("label", ""),
            }
            for nid in sorted(record["nodes"])
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def compute_hierarchy_layout(record: Dict[str, Any]) -> Dict[str, Any]:
    node_widths: Dict[str, float] = {}
    subtree_widths: Dict[str, float] = {}
    node_lefts: Dict[str, float] = {}
    rows_by_depth: Dict[int, List[str]] = {}

    def measure(node_id: str) -> float:
        own_width = hierarchy_button_width_px(node_id)
        node_widths[node_id] = own_width

        children = tree_display_children(record, node_id)
        if not children:
            subtree_width = own_width
        else:
            children_total = 0.0
            for idx, child_id in enumerate(children):
                children_total += measure(child_id)
                if idx < len(children) - 1:
                    children_total += TREE_SIBLING_GAP_PX

            # 부모 버튼이 더 길면 그 길이까지 subtree 폭 확보
            subtree_width = max(own_width, children_total)

        subtree_widths[node_id] = subtree_width
        return subtree_width

    def place(node_id: str, left_x: float, depth: int) -> None:
        # 부모의 왼쪽 시작 = 가장 왼쪽 자식 subtree 시작
        node_lefts[node_id] = left_x
        rows_by_depth.setdefault(depth, []).append(node_id)

        children = tree_display_children(record, node_id)
        if not children:
            return

        child_left = left_x
        for idx, child_id in enumerate(children):
            place(child_id, child_left, depth + 1)
            child_left += subtree_widths[child_id]
            if idx < len(children) - 1:
                child_left += TREE_SIBLING_GAP_PX

    # root_ids = record["roots"]
    root_ids = display_root_ids(record)
    if not root_ids:
        return {
            "rows": [],
            "node_lefts": {},
            "node_widths": {},
            "tree_width": 0.0,
        }

    total_width = TREE_SIDE_PAD_PX * 2.0
    for idx, root_id in enumerate(root_ids):
        total_width += measure(root_id)
        if idx < len(root_ids) - 1:
            total_width += TREE_ROOT_GAP_PX

    left_x = TREE_SIDE_PAD_PX
    for idx, root_id in enumerate(root_ids):
        place(root_id, left_x, 0)
        left_x += subtree_widths[root_id]
        if idx < len(root_ids) - 1:
            left_x += TREE_ROOT_GAP_PX

    ordered_rows: List[List[str]] = []
    for depth in sorted(rows_by_depth):
        ordered_rows.append(sorted(rows_by_depth[depth], key=lambda nid: node_lefts[nid]))

    return {
        "rows": ordered_rows,
        "node_lefts": node_lefts,
        "node_widths": node_widths,
        "tree_width": total_width,
    }

def build_tree_connector_svg(
    record: Dict[str, Any],
    rows: List[List[str]],
    node_lefts: Dict[str, float],
    node_widths: Dict[str, float],
    tree_width: float,
) -> str:
    if not rows:
        return ""

    svg_height = (
        TREE_PANEL_TOP_PAD_PX * 2
        + len(rows) * TREE_ROW_HEIGHT_PX
        + max(0, len(rows) - 1) * TREE_ROW_GAP_PX
    )

    def row_center_y(row_idx: int) -> float:
        return (
            TREE_PANEL_TOP_PAD_PX
            + row_idx * (TREE_ROW_HEIGHT_PX + TREE_ROW_GAP_PX)
            + TREE_ROW_HEIGHT_PX / 2.0
        )

    def node_center_x(node_id: str) -> float:
        return node_lefts[node_id] + node_widths[node_id] / 2.0

    strokes: List[str] = []

    for row_idx, parent_row in enumerate(rows[:-1]):
        child_row = rows[row_idx + 1]
        parent_y = row_center_y(row_idx)
        child_y = row_center_y(row_idx + 1)

        bus_y = child_y - TREE_ROW_HEIGHT_PX / 2.0 - 6.0

        child_set = set(child_row)

        for parent_id in parent_row:
            children = [c for c in tree_display_children(record, parent_id) if c in child_set]
            if not children:
                continue

            px = node_center_x(parent_id)

            if len(children) == 1:
                cx = node_center_x(children[0])
                # parent -> child 직접 연결
                strokes.append(
                    f"<line x1='{px:.1f}' y1='{parent_y + TREE_ROW_HEIGHT_PX/2 - 6:.1f}' "
                    f"x2='{cx:.1f}' y2='{child_y - TREE_ROW_HEIGHT_PX/2 + 6:.1f}' />"
                )
                continue

            child_centers = [node_center_x(c) for c in children]
            left_x = min(child_centers)
            right_x = max(child_centers)

            # parent 아래로
            strokes.append(
                f"<line x1='{px:.1f}' y1='{parent_y + TREE_ROW_HEIGHT_PX/2 - 6:.1f}' "
                f"x2='{px:.1f}' y2='{bus_y:.1f}' />"
            )

            # 형제 가로선
            strokes.append(
                f"<line x1='{left_x:.1f}' y1='{bus_y:.1f}' "
                f"x2='{right_x:.1f}' y2='{bus_y:.1f}' />"
            )

            # 각 child로 내려가는 선
            for cx in child_centers:
                strokes.append(
                    f"<line x1='{cx:.1f}' y1='{bus_y:.1f}' "
                    f"x2='{cx:.1f}' y2='{child_y - TREE_ROW_HEIGHT_PX/2 + 6:.1f}' />"
                )

    svg = f"""
    <div style="position:relative; height:0; overflow:visible; pointer-events:none;">
    <svg
        width="{int(math.ceil(tree_width))}"
        height="{int(math.ceil(svg_height))}"
        viewBox="0 0 {int(math.ceil(tree_width))} {int(math.ceil(svg_height))}"
        xmlns="http://www.w3.org/2000/svg"
        style="position:absolute; left:0; top:0; z-index:0; pointer-events:none; overflow:visible;">
        <g stroke="rgba(148,163,184,0.55)" stroke-width="2" fill="none" stroke-linecap="round">
            {''.join(strokes)}
        </g>
    </svg>
    </div>
    """
    return svg

def build_row_parts_from_layout(
    row: List[str],
    node_lefts: Dict[str, float],
    node_widths: Dict[str, float],
    tree_width: float,
) -> List[Tuple[str, float, Optional[str]]]:
    parts: List[Tuple[str, float, Optional[str]]] = []
    prev_end = 0.0

    row = sorted(row, key=lambda nid: node_lefts[nid])

    for node_id in row:
        start_x = node_lefts[node_id]
        width_px = node_widths[node_id]

        spacer_width = start_x - prev_end
        if spacer_width > 0.5:
            parts.append(("spacer", spacer_width, None))

        parts.append(("button", width_px, node_id))
        prev_end = start_x + width_px

    # trailing_width = tree_width - prev_end
    # if trailing_width > 0.5:
    #     parts.append(("spacer", trailing_width, None))

    return parts

# Dataset parsing

def build_tree(image_id: str, node_folder_names: List[str], image_dir: Path) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}

    def ensure_node(node_id: str, actual: bool = False, folder_name: Optional[str] = None) -> None:
        if node_id not in nodes:
            parent = None
            if "__" in node_id:
                parent = node_id.rsplit("__", 1)[0]
            nodes[node_id] = {
                "id": node_id,
                "label": human_label(node_id),
                "parent": parent,
                "children": [],
                "actual": actual,
                "folder_name": folder_name,
            }
        else:
            if actual:
                nodes[node_id]["actual"] = True
                nodes[node_id]["folder_name"] = folder_name

    for folder in sorted(set(node_folder_names)):
        if folder == "__subcrops":
            continue
        parts = folder.split("__")
        for idx in range(1, len(parts) + 1):
            node_id = "__".join(parts[:idx])
            is_actual = idx == len(parts)
            ensure_node(node_id, actual=is_actual, folder_name=folder if is_actual else None)

    for node_id, node in nodes.items():
        parent = node["parent"]
        if parent and parent in nodes and node_id not in nodes[parent]["children"]:
            nodes[parent]["children"].append(node_id)

    for node in nodes.values():
        node["children"] = sorted(node["children"], key=lambda x: (x.count("__"), x))

    root_candidates = sorted([nid for nid, info in nodes.items() if info["parent"] is None])
    actual_nodes = [nid for nid, info in nodes.items() if info["actual"]]

    return {
        "image_id": image_id,
        "image_dir": str(image_dir),
        "nodes": nodes,
        "roots": root_candidates,
        "actual_nodes": sorted(actual_nodes),
    }

# def scan_dataset(dataset_root: Path) -> Dict[str, Any]:
#     records: Dict[str, Any] = {}
#     if not dataset_root.exists() or not dataset_root.is_dir():
#         return records

#     for image_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
#         node_folders = [
#             p.name
#             for p in image_dir.iterdir()
#             if p.is_dir() and not p.name.startswith(".") and p.name != "__subcrops"
#         ]
#         if not node_folders:
#             continue
#         records[image_dir.name] = build_tree(image_dir.name, node_folders, image_dir=image_dir)
#     return records

@st.cache_data(ttl=300, show_spinner=False)
def load_records_for_reviewer(reviewer_id: str) -> Dict[str, Any]:
    sb = get_supabase_client()

    assign_resp = (
        sb.table("review_assignments")
        .select("image_id, sort_index")
        .eq("reviewer_id", reviewer_id)
        .order("sort_index")
        .execute()
    )
    assignments = assign_resp.data or []
    image_ids = [r["image_id"] for r in assignments]

    if not image_ids:
        return {}

    manifest_resp = (
        sb.table("image_manifest")
        .select("*")
        .in_("image_id", image_ids)
        .execute()
    )
    rows = manifest_resp.data or []

    by_id = {row["image_id"]: build_record_from_manifest(row) for row in rows}
    return {image_id: by_id[image_id] for image_id in image_ids if image_id in by_id}


def build_record_from_manifest(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "dataset_prefix": row["dataset_prefix"],
        "root_image_path": row["root_image_path"],
        "root_overlay_path": row["root_overlay_path"],
        "full_size": row["full_size"],
        "nodes": row["nodes"],
        "roots": row["roots"],
        "actual_nodes": row["actual_nodes"],
    }

@st.cache_data(show_spinner=False)
def load_all_records(dataset_root_str: str) -> Dict[str, Any]:
    return scan_dataset(Path(dataset_root_str))

def filter_records_by_reviewer(records: Dict[str, Any], reviewer_id: str) -> Dict[str, Any]:
    reviewer_id = (reviewer_id or "").strip()
    user_cfg = REVIEW_USERS.get(reviewer_id)
    if not user_cfg:
        return {}

    ranges = user_cfg["ranges"]

    sorted_ids = sorted(records.keys())
    selected_ids: List[str] = []
    for start, end in ranges:
        selected_ids.extend(sorted_ids[start:end])

    seen = set()
    ordered_unique_ids = []
    for image_id in selected_ids:
        if image_id not in seen and image_id in records:
            seen.add(image_id)
            ordered_unique_ids.append(image_id)

    return {image_id: records[image_id] for image_id in ordered_unique_ids}

# Annotation state

def ensure_annotation_bucket(image_id: str) -> Dict[str, Any]:
    annotations = st.session_state.annotations
    if image_id not in annotations:
        annotations[image_id] = {
            "nodes": {},
            "tree_summary": {"answers": {}, "updated_at": None},
            "image_review": {"updated_at": None},
        }
    return annotations[image_id]

def ensure_node_bucket(image_id: str, node_id: str) -> Dict[str, Any]:
    image_bucket = ensure_annotation_bucket(image_id)
    if node_id not in image_bucket["nodes"]:
        image_bucket["nodes"][node_id] = {"answers": {}, "updated_at": None,}
    return image_bucket["nodes"][node_id]

def current_annotation_path() -> Path:
    reviewer = safe_token(st.session_state.reviewer_id.strip() or "anonymous")
    return OUTPUT_ROOT / f"annotations_{reviewer}.json"

def persist_annotations() -> None:
    save_json(current_annotation_path(), st.session_state.annotations)

def load_annotations_if_needed() -> None:
    reviewer = safe_token(st.session_state.reviewer_id.strip() or "anonymous")
    loaded_key = f"db::{reviewer}"

    if st.session_state.get("_loaded_annotation_path") != loaded_key:
        st.session_state.annotations = load_annotations_from_supabase(reviewer)
        st.session_state._loaded_annotation_path = loaded_key
        st.session_state["annotations_dirty"] = False


# Questions

def node_questions_for(node_id: str) -> List[Dict[str, Any]]:
    leaf = human_label(node_id).replace("_", " ")
    return [
        {"id": "presence", "label": f"Q1. <{leaf}> 노드가 실제 대상과 맞나요?", "type": "single_choice", "options": ["예", "아니오", "애매함"], "required": True},
        {"id": "boundary", "label": "Q2. 마스크 경계가 깔끔한가요?", "type": "single_choice", "options": ["매우 좋음", "보통", "나쁨"], "required": True},
        {"id": "missing_area", "label": "Q3. 누락된 영역이 있나요?", "type": "single_choice", "options": ["없음", "조금 있음", "많음"], "required": True},
        {"id": "extra_area", "label": "Q4. 과하게 포함된 영역이 있나요?", "type": "single_choice", "options": ["없음", "조금 있음", "많음"], "required": True},
        {"id": "visibility", "label": "Q5. 가려짐/잘림 때문에 평가가 어렵나요?", "type": "single_choice", "options": ["아니오", "조금", "많이"], "required": True},
        {"id": "score", "label": "Q6. 전체 품질 점수", "type": "single_choice", "options": ["1", "2", "3", "4", "5"], "required": True},
        {"id": "issue_tags", "label": "Q7. 이슈 태그", "type": "multi_choice", "options": ["boundary", "missing", "extra", "wrong_class", "tiny_object", "occlusion"], "required": False},
        {"id": "comment", "label": "Q8. 메모", "type": "text", "required": False},
    ]

def tree_questions_for(image_id: str) -> List[Dict[str, Any]]:
    return [
        {"id": "overall_consistency", "label": "전체 트리 구조가 일관적인가요?", "type": "single_choice", "options": ["예", "아니오", "애매함"], "required": True},
        {"id": "missing_critical_nodes", "label": "치명적으로 빠진 노드가 있나요?", "type": "single_choice", "options": ["없음", "있음"], "required": True},
        {"id": "ontology_fit", "label": "현재 taxonomy / ontology가 장면을 잘 설명하나요?", "type": "single_choice", "options": ["좋음", "보통", "나쁨"], "required": True},
        {"id": "priority_fix", "label": "가장 먼저 수정해야 할 우선순위", "type": "single_choice", "options": ["없음", "라벨 수정", "마스크 수정", "노드 추가/삭제", "taxonomy 수정"], "required": True},
        {"id": "summary_comment", "label": "전체 트리 메모", "type": "text", "required": False},
    ]


# Completion logic

def get_answers_bucket(image_id: str, mode: str, node_id: Optional[str] = None) -> Dict[str, Any]:
    if mode == "node":
        return ensure_node_bucket(image_id, node_id or "")
    return ensure_annotation_bucket(image_id)["tree_summary"]

def required_missing_questions(image_id: str, mode: str, questions: List[Dict[str, Any]], node_id: Optional[str] = None) -> List[str]:
    answers = get_answers_bucket(image_id, mode, node_id).get("answers", {})
    missing: List[str] = []
    for q in questions:
        if not q.get("required", False):
            continue
        value = answers.get(q["id"])
        if q["type"] == "multi_choice":
            if not value:
                missing.append(q["id"])
        elif value in (None, "", []):
            missing.append(q["id"])
    return missing

def node_confirmed(image_id: str, node_id: str) -> bool:
    if not is_reviewable_node(node_id):
        return True
    return len(required_missing_questions(image_id, "node", node_questions_for(node_id), node_id=node_id,)) == 0

def all_nodes_confirmed(image_id: str, record: Dict[str, Any]) -> bool:
    actual_nodes = [node_id for node_id in record["actual_nodes"] if is_reviewable_node(node_id)]
    return bool(actual_nodes) and all(node_confirmed(image_id, node_id) for node_id in actual_nodes)

def tree_summary_confirmed(image_id: str) -> bool:
    return len(required_missing_questions(image_id, "tree", tree_questions_for(image_id),)) == 0

def image_complete(image_id: str, record: Dict[str, Any]) -> bool:
    return all_nodes_confirmed(image_id, record) and tree_summary_confirmed(image_id)

def node_progress(image_id: str, record: Dict[str, Any]) -> Tuple[int, int]:
    actual_nodes = [node_id for node_id in record["actual_nodes"] if is_reviewable_node(node_id)]
    done = sum(1 for node_id in actual_nodes if node_confirmed(image_id, node_id))
    total = len(actual_nodes)
    return done, total

def missing_report(image_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    missing_nodes: List[Dict[str, Any]] = []

    for node_id in record["actual_nodes"]:
        if not is_reviewable_node(node_id):
            continue
        q_missing = required_missing_questions(image_id, "node", node_questions_for(node_id), node_id=node_id,)
        if q_missing:
            missing_nodes.append({"node_id": node_id, "missing_question_ids": q_missing,})
    tree_missing = required_missing_questions(image_id, "tree", tree_questions_for(image_id),)

    return {"missing_nodes": missing_nodes, "tree_missing": tree_missing,}

# Asset lookup
def get_inspector_pills(record: Dict[str, Any], node_id: str) -> List[str]:
    node = record["nodes"].get(node_id)
    if node is None:
        return ["현재: -", "부모: -", "자식: -", "깊이: -", "경로: -"]

    parent_id = node.get("parent")
    depth = get_node_depth(node_id)

    children = [
        child_id
        for child_id in node["children"]
        if not (record["nodes"][child_id]["actual"] and not is_reviewable_node(child_id))
    ]

    parent_label = human_label(parent_id) if parent_id else "-"
    child_labels = [human_label(child_id) for child_id in children]

    if child_labels:
        if len(child_labels) > 5:
            children_text = ", ".join(child_labels[:5]) + f" 외 {len(child_labels) - 5}개"
        else:
            children_text = ", ".join(child_labels)
    else:
        children_text = "-"

    pretty_path = " → ".join(get_node_path_labels(node_id))

    return [
        f"현재: {human_label(node_id)}",
        f"부모: {parent_label}",
        f"자식: {children_text}",
        f"깊이: {depth}",
        f"경로: {pretty_path}",
    ]

def node_assets(record: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    node = record["nodes"].get(node_id, {})
    bbox = node.get("bbox")

    return {
        "root_original": record.get("root_image_path"),
        "root_overlay": record.get("root_overlay_path"),
        "mask_original": node.get("mask_original_path"),
        "mask": node.get("mask_path"),
        "instances": node.get("instance_paths", []) or [],
        "full_size": tuple(record["full_size"]) if record.get("full_size") else None,
        "bbox": tuple(bbox) if bbox else None,
    }


# Styling

def inject_box_style(box_key: str, *, selected: bool, done: bool, muted: bool = False) -> None:
    if selected:
        bg = "rgba(59,130,246,0.10)"       # blue
        border = "#2563eb"
    elif done:
        bg = "rgba(34,197,94,0.12)"        # green
        border = "#16a34a"
    else:
        bg = "rgba(255,255,255,0.02)"
        border = "rgba(148,163,184,0.30)"

    opacity = "0.55" if muted else "1"

    st.markdown(
        f"""
        <style>
        .st-key-{box_key} {{
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.35rem 0.45rem 0.15rem 0.45rem;
            margin-bottom: 0.35rem;
            background: {bg};
            opacity: {opacity};
        }}
        .st-key-{box_key} button {{
            justify-content: flex-start;
            text-align: left;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 3rem; padding-bottom: 1rem;}
        .small-muted { color: rgba(148,163,184,1); font-size: 0.86rem; }
        .section-card {
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            background: rgba(255,255,255,0.02);
        }
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.9rem;
            border: 1px solid rgba(148,163,184,0.25);
            margin-right: 0.35rem;
        }
        .image-item.done button {
            background-color: #dbeafe !important;
            color: #1e3a8a !important;
        }
        .auth-card {
            width: 100%;
            max-width: 520px;
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 20px;
            padding: 2rem 2rem 1.4rem 2rem;
            background: rgba(255,255,255,0.03);
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        }
        .auth-badge {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            color: #bfdbfe;
            background: rgba(59,130,246,0.12);
            border: 1px solid rgba(59,130,246,0.28);
            margin-bottom: 0.9rem;
        }
        .auth-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            line-height: 1.2;
        }
        .auth-subtitle {
            color: rgba(148,163,184,1);
            font-size: 0.96rem;
            margin-bottom: 1.2rem;
        }
        .auth-help {
            margin-top: 1rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(148,163,184,0.16);
            color: rgba(148,163,184,1);
            font-size: 0.84rem;
        }
        textarea {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        div[data-testid="stCheckbox"] {
            margin-bottom: -6px;
        }
        div[role="radiogroup"] {
            gap: 0.15rem !important;
        }
        .image-item button {
            height: 2.2rem;
            padding: 0.2rem 0.5rem;
            font-size: 0.85rem;
            border-radius: 0.5rem;
        }
        .image-item.done button {
            background-color: #dbeafe !important;
            color: #1e3a8a !important;
        }
        [data-testid="stImage"] img {
            border: 1px solid rgba(148,163,184,0.28);
            border-radius: 6px;
        }
        .finalize-pill {
            display: inline-block;
            padding: 0.16rem 0.52rem;
            border-radius: 999px;
            font-size: 0.78rem;
            color: rgba(148,163,184,0.92);
            background: rgba(148,163,184,0.08);
            border: 1px solid rgba(148,163,184,0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
def inject_hierarchy_compact_css(panel_key: str) -> None:
    panel_key = safe_token(panel_key)
    st.markdown(
        f"""
        <style>
        .st-key-{panel_key} [data-testid="stElementContainer"] {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }}
        .st-key-{panel_key} [data-testid="stMarkdownContainer"] p {{
            margin: 0 !important;
            line-height: 0 !important;
        }}
        .st-key-{panel_key} [data-testid="stButton"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-{panel_key} [data-testid="column"] {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}
        .st-key-{panel_key} [data-testid="stVerticalBlock"] {{
            gap: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_hierarchy_panel_style(panel_key: str) -> None:
    panel_key = safe_token(panel_key)

    st.markdown(
        f"""
        <style>
        .st-key-{panel_key} {{
            position: relative;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 0 0 10px 0 !important;
            margin: 0 !important;
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 12px;
            background: rgba(255,255,255,0.01);
        }}

        .st-key-{panel_key} > div {{
            position: relative;
            z-index: 1;
            padding: 0 !important;
            margin: 0 !important;
        }}

        .st-key-{panel_key} [data-testid="stVerticalBlock"] {{
            width: max-content !important;
            min-width: max-content !important;
            gap: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_hierarchy_row_style(row_key: str, total_width_px: float) -> None:
    row_key = safe_token(row_key)
    total_width_px = int(math.ceil(total_width_px))

    st.markdown(
        f"""
        <style>
        .st-key-{row_key} {{
            position: relative;
            z-index: 2;
            height: {TREE_ROW_HEIGHT_PX}px;
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{row_key} > div {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{row_key} [data-testid="stHorizontalBlock"] {{
            width: {total_width_px}px !important;
            min-width: {total_width_px}px !important;
            gap: 0 !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{row_key} [data-testid="column"] {{
            padding: 0 !important;
            margin: 0 !important;
        }}

        .st-key-{row_key} [data-testid="stMarkdownContainer"] p {{
            margin: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_hierarchy_button_style(
    button_key: str,
    *,
    selected: bool,
    done: bool,
) -> None:
    safe_key = safe_token(button_key)

    if selected:
        bg = "rgba(59,130,246,0.16)"
        border = "#2563eb"
        text = "#dbeafe"
    elif done:
        bg = "rgba(34,197,94,0.14)"
        border = "#16a34a"
        text = "#dcfce7"
    else:
        bg = "rgba(255,255,255,0.03)"
        border = "rgba(148,163,184,0.35)"
        text = "inherit"

    st.markdown(
        f"""
        <style>
        .st-key-{safe_key} {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{safe_key} > div {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{safe_key} [data-testid="stButton"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .st-key-{safe_key} button {{
            background: {bg} !important;
            border: 1px solid {border} !important;
            color: {text} !important;
            height: 30px !important;
            border-radius: 999px !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            white-space: nowrap !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
            text-align: center !important;
            justify-content: center !important;
            line-height: 1 !important;
            padding: 0 0.4rem !important;
            box-shadow: none !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Widget sync helpers

def question_widget_key(image_id: str, mode: str, qid: str, node_id: Optional[str] = None) -> str:
    if mode == "node":
        return f"q__{safe_token(image_id)}__{safe_token(node_id or '')}__{safe_token(qid)}"
    return f"q__{safe_token(image_id)}__tree__{safe_token(qid)}"

def init_widget_value(widget_key: str, q: Dict[str, Any], stored_value: Any) -> None:
    if widget_key in st.session_state:
        return
    qtype = q["type"]
    if qtype == "single_choice":
        st.session_state[widget_key] = stored_value if stored_value is not None else None
    elif qtype == "multi_choice":
        st.session_state[widget_key] = stored_value if stored_value is not None else []
    elif qtype == "text":
        st.session_state[widget_key] = stored_value if stored_value is not None else ""
    else:
        st.session_state[widget_key] = stored_value

def pull_widget_value(widget_key: str, q: Dict[str, Any]) -> Any:
    raw = st.session_state.get(widget_key)
    if q["type"] == "single_choice":
        return raw
    if q["type"] == "multi_choice":
        return raw or []
    if q["type"] == "text":
        return (raw or "").strip()
    return raw


def sync_questions_to_bucket(image_id: str, mode: str, questions: List[Dict[str, Any]], node_id: Optional[str] = None,) -> None:
    bucket = get_answers_bucket(image_id, mode, node_id)
    answers = bucket.setdefault("answers", {})
    changed = False

    for q in questions:
        widget_key = question_widget_key(image_id, mode, q["id"], node_id)
        if widget_key not in st.session_state:
            continue

        new_value = pull_widget_value(widget_key, q)
        old_value = answers.get(q["id"])

        if old_value != new_value:
            answers[q["id"]] = new_value
            changed = True

    if changed:
        bucket["updated_at"] = now_iso()
        st.session_state["annotations_dirty"] = True

# Rendering: navigation

def render_image_list(records: Dict[str, Any]) -> None:
    st.subheader("Images")
    search = st.text_input("검색", key="image_search", placeholder="image id 검색")
    filtered_ids = [image_id for image_id in sorted(records) if search.lower() in image_id.lower()]
    st.caption(f"{len(filtered_ids)}개 표시")

    with st.container(height=960):
        for image_id in filtered_ids:
            record = records[image_id]
            # done, total = node_progress(image_id, record)
            # selected = st.session_state.selected_image_id == image_id
            # completed = image_complete(image_id, record)

            selected = st.session_state.selected_image_id == image_id
            done, total = node_progress(image_id, record)

            if selected or done > 0:
                completed = image_complete(image_id, record)
            else:
                completed = False


            box_key = f"imgbox_{safe_token(image_id)}"
            inject_box_style(box_key, selected=selected, done=completed)
            with st.container(key=box_key):
                icon = "🟦" if completed else ("🟡" if done else "⬜")
                display_id = str(int(image_id)) if str(image_id).isdigit() else str(image_id)
                if st.button(f"{icon} {display_id}", key=f"pick_image_{image_id}", use_container_width=True):
                    st.session_state.selected_image_id = image_id
                    st.session_state.selected_mode = "node"
                    first_node = first_reviewable_node_id(records[image_id])
                    st.session_state.selected_node_id = first_node
                    st.rerun()
                progress = (done / total * 100) if total > 0 else 0
                st.caption(f"노드 진행률: {done}/{total} • {progress:.1f}% 완료")
                
def render_tree_panel(record: Optional[Dict[str, Any]]) -> None:
    st.subheader("Tree")
    if record is None:
        st.info("왼쪽에서 image를 선택하세요.")
        return

    image_id = record["image_id"]
    done, total = node_progress(image_id, record)
    st.markdown(
        f"<div class='section-card'><b>{image_id}</b><br/>노드 완료: {done}/{total}<br/>전체 트리 질문: {'완료' if tree_summary_confirmed(image_id) else '미완료'}</div>",
        unsafe_allow_html=True,
    )

    tree_done = tree_summary_confirmed(image_id)
    tree_enabled = all_nodes_confirmed(image_id, record)
    box_key = f"treectrl_{safe_token(image_id)}"
    inject_box_style(box_key, selected=st.session_state.selected_mode == "tree", done=tree_done, muted=not tree_enabled)
    with st.container(key=box_key):
        if st.button(f"{'🟦' if tree_done else '⬜'} 전체 트리 질문", key=f"open_tree_summary_{image_id}", use_container_width=True, disabled=not tree_enabled):
            st.session_state.selected_mode = "tree"
            st.session_state.selected_node_id = None
            # st.rerun()
        if not tree_enabled:
            st.caption("모든 노드를 완료해야 활성화됩니다.")

    with st.container(height=650):
        for root_id in record["roots"]:
            render_tree_node(record, root_id, depth=0)

def render_tree_node(record: Dict[str, Any], node_id: str, depth: int) -> None:
    node = record["nodes"][node_id]

    if node["actual"] and not is_reviewable_node(node_id):
        for child_id in node["children"]:
            render_tree_node(record, child_id, depth)
        return

    image_id = record["image_id"]
    selected = st.session_state.selected_mode == "node" and st.session_state.selected_node_id == node_id
    done = node_confirmed(image_id, node_id) if node["actual"] else False
    is_clickable = node["actual"]

    box_key = f"nodebox_{safe_token(image_id)}_{safe_token(node_id)}"
    inject_box_style(box_key, selected=selected, done=done, muted=not is_clickable)

    indent = "　" * depth
    icon = "🟦" if done else ("⬜" if is_clickable else "◽")
    prefix = "▾ " if node["children"] else "• "
    label = f"{indent}{icon} {prefix}{node['label']}"

    with st.container(key=box_key):
        if is_clickable:
            if st.button(label, key=f"pick_node_{image_id}_{node_id}", use_container_width=True):
                st.session_state.selected_image_id = image_id
                st.session_state.selected_mode = "node"
                st.session_state.selected_node_id = node_id
                # st.rerun()
        else:
            st.markdown(f"<div class='small-muted'>{label}</div>", unsafe_allow_html=True)

        if is_clickable:
            missing = required_missing_questions(image_id, "node", node_questions_for(node_id), node_id=node_id)
            status_txt = "완료" if done else f"미답변 {len(missing)}개"
            st.caption(f"{node_id} | {status_txt}")
        else:
            st.caption(f"synthetic parent | {node_id}")

    for child_id in node["children"]:
        render_tree_node(record, child_id, depth + 1)

# Rendering: hierarchy
def row_render_width(row: List[str], node_lefts: Dict[str, float], node_widths: Dict[str, float]) -> float:
    if not row:
        return TREE_SIDE_PAD_PX * 2
    last = max(row, key=lambda nid: node_lefts[nid] + node_widths[nid])
    return node_lefts[last] + node_widths[last] + TREE_SIDE_PAD_PX
def render_experimental_tree_panel(record: Dict[str, Any]) -> None:
    if record is None:
        return

    image_id = record["image_id"]

    # record_sig = record_structure_signature(record)
    record_for_layout = inject_tree_summary_node(record)
    record_sig = record_structure_signature(record_for_layout)

    layout = compute_hierarchy_layout_cached(record_sig)
    rows = layout["rows"]
    node_lefts = layout["node_lefts"]
    node_widths = layout["node_widths"]
    tree_width = layout["tree_width"]

    tree_done = tree_summary_confirmed(image_id)
    tree_enabled = all_nodes_confirmed(image_id, record)
    tree_selected = st.session_state.selected_mode == "tree"

    tree_summary_width = max(
        TREE_BUTTON_MIN_WIDTH_PX,
        min(TREE_BUTTON_MAX_WIDTH_PX, TREE_BUTTON_BASE_PX + TREE_CHAR_WIDTH_PX * len("전체트리"))
    )

    if not rows:
        return
    
    display_rows = rows

    st.markdown("#### Hierarchy View")


    with st.container():
        panel_key = safe_token(f"hier_panel__{record['image_id']}")
        inject_hierarchy_panel_style(panel_key)
        inject_hierarchy_compact_css(panel_key)

        with st.container(key=panel_key):
            # node_children_map = {nid: record["nodes"][nid]["children"] for nid in record["nodes"]}
            node_children_map = {nid: record_for_layout["nodes"][nid]["children"] for nid in record_for_layout["nodes"]}

            node_actual_map = {nid: record_for_layout["nodes"][nid]["actual"] for nid in record_for_layout["nodes"]}

            svg_html = build_tree_connector_svg_cached(
                record["image_id"],
                rows,
                node_lefts,
                node_widths,
                tree_width,
                node_children_map,
                node_actual_map,
            )
            st.markdown(svg_html, unsafe_allow_html=True)

                    # row=[nid for nid in row if nid != TREE_SUMMARY_NODE_ID],
            for row_idx, row in enumerate(display_rows):
                parts = build_row_parts_from_layout(
                    row=row,
                    node_lefts=node_lefts,
                    node_widths=node_widths,
                    tree_width=tree_width,
                )

                spec = [max(1.0, width_px) for _, width_px, _ in parts]

                row_key = safe_token(f"hier_row__{record['image_id']}__{row_idx}")
                row_width = row_render_width(row, node_lefts, node_widths)
                inject_hierarchy_row_style(row_key, row_width)
                    
                # inject_hierarchy_row_style(row_key, tree_width)

                with st.container(key=row_key):
                    cols = st.columns(spec, gap=None)

                    for part_idx, (col, (kind, _width_px, node_id)) in enumerate(zip(cols, parts)):
                        with col:
                            if kind == "spacer" or node_id is None:
                                st.empty()
                                continue

                            # special root node: 전체트리
                            if node_id == TREE_SUMMARY_NODE_ID:
                                label = "전체트리"
                                selected = st.session_state.selected_mode == "tree"
                                done = tree_summary_confirmed(record["image_id"])
                                is_clickable = all_nodes_confirmed(record["image_id"], record)

                                button_key = safe_token(
                                    f"exp_tree_btn__{record['image_id']}__tree_summary__r{row_idx}__p{part_idx}"
                                )

                                inject_hierarchy_button_style(
                                    button_key=button_key,
                                    selected=selected,
                                    done=done,
                                )

                                if st.button(
                                    label,
                                    key=button_key,
                                    use_container_width=True,
                                    disabled=not is_clickable,
                                ):
                                    st.session_state.selected_mode = "tree"
                                    st.session_state.selected_node_id = None
                                    st.rerun()

                                continue

                            # 일반 노드
                            is_actual = record["nodes"][node_id]["actual"]
                            is_reviewable = is_reviewable_node(node_id)
                            is_clickable = is_actual and is_reviewable

                            label = human_label(node_id)
                            selected = (
                                st.session_state.selected_mode == "node"
                                and st.session_state.selected_node_id == node_id
                            )
                            done = node_confirmed(record["image_id"], node_id) if is_actual else False

                            button_key = safe_token(
                                f"exp_tree_btn__{record['image_id']}__{node_id}__r{row_idx}__p{part_idx}"
                            )

                            inject_hierarchy_button_style(
                                button_key=button_key,
                                selected=selected,
                                done=done,
                            )

                            if is_clickable:
                                if st.button(label, key=button_key, use_container_width=True):
                                    st.session_state.selected_image_id = record["image_id"]
                                    st.session_state.selected_mode = "node"
                                    st.session_state.selected_node_id = node_id
                                    st.rerun()
                            else:
                                st.button(label, key=button_key, use_container_width=True, disabled=True)

def render_visual_header(record: Dict[str, Any], node_id: str) -> None:
    pills = get_inspector_pills(record, node_id)

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.35rem;">
            <div style="font-size:1.5rem; font-weight:700;">Visuals</div>
            <div style="display:flex; align-items:center; gap:0.35rem; flex-wrap:wrap;">
                {''.join([f"<span class='status-pill'>{p}</span>" for p in pills])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# def render_asset_panel(record: Dict[str, Any], node_id: str) -> None:
#     render_visual_header(record, node_id)

#     assets = node_assets(record, node_id)
#     leaf = human_label(node_id)

#     tabs = st.tabs(["기본", "인스턴스"])

#     with tabs[0]:
#         review_items: List[Image.Image] = []
#         review_captions: List[str] = []

#         # 1) 원본 이미지
#         if assets["root_original"]:
#             img = build_direct_display(assets["root_original"])
#             if img is not None:
#                 review_items.append(img)
#                 review_captions.append("원본 이미지")

#         # 2) 원본 이미지 + 노드 강조
#         overlay_img = build_overlay_on_full_image(record, node_id)
#         if overlay_img is not None:
#             review_items.append(overlay_img)
#             review_captions.append(f"<{leaf}> 오버레이")

#         # 3) 노드 영역 원본
#         if assets["mask_original"]:
#             img = build_mask_original_display(record, node_id, assets["mask_original"])
#             if img is not None:
#                 review_items.append(img)
#                 review_captions.append(f"<{leaf}> 원본")

#         # 4) 노드 마스크
#         if assets["mask"]:
#             img = build_direct_display(assets["mask"])
#             if img is not None:
#                 review_items.append(img)
#                 review_captions.append(f"<{leaf}> 마스크")

#         if review_items:
#             # st.image(review_items, caption=review_captions, width="content")
#             st.image(review_items, caption=review_captions, width=320)
#         else:
#             st.info("노드 검토용 이미지를 찾지 못했습니다.")

#     with tabs[1]:
#         if assets["instances"]:
#             imgs: List[Image.Image] = []
#             caps: List[str] = []
#             for i, p in enumerate(assets["instances"], start=1):
#                 img = build_direct_display(p)
#                 if img is not None:
#                     imgs.append(img)
#                     caps.append(f"{leaf} #{i}")

#             if imgs:
#                 # st.image(imgs, caption=caps, width="content")
#                 st.image(imgs, caption=caps, width=320)
#             else:
#                 st.info("인스턴스 마스크 이미지를 읽지 못했습니다.")
#         else:
#             st.info("인스턴스 마스크가 없습니다.")
def render_asset_panel(record: Dict[str, Any], node_id: str) -> None:
    render_visual_header(record, node_id)

    assets = node_assets(record, node_id)
    leaf = human_label(node_id)

    # 기본 이미지만 바로 표시
    review_items: List[Image.Image] = []
    review_captions: List[str] = []

    # 1) 원본 이미지
    if assets["root_original"]:
        img = build_direct_display(assets["root_original"])
        if img is not None:
            review_items.append(img)
            review_captions.append("원본 이미지")

    # 2) 원본 이미지 + 노드 강조
    overlay_img = build_overlay_on_full_image(record, node_id)
    if overlay_img is not None:
        review_items.append(overlay_img)
        review_captions.append(f"<{leaf}> 오버레이")

    # 3) 노드 영역 원본
    if assets["mask_original"]:
        img = build_mask_original_display(record, node_id, assets["mask_original"])
        if img is not None:
            review_items.append(img)
            review_captions.append(f"<{leaf}> 원본")

    # 4) 노드 마스크
    if assets["mask"]:
        img = build_direct_display(assets["mask"])
        if img is not None:
            review_items.append(img)
            review_captions.append(f"<{leaf}> 마스크")

    if review_items:
        st.image(review_items, caption=review_captions, width=320)
    else:
        st.info("노드 검토용 이미지를 찾지 못했습니다.")

    # 인스턴스 이미지는 필요할 때만 열기
    show_instances = st.toggle(
        "인스턴스 보기",
        value=False,
        key=f"show_instances_{record['image_id']}_{node_id}",
    )

    if show_instances:
        if assets["instances"]:
            imgs: List[Image.Image] = []
            caps: List[str] = []

            for i, p in enumerate(assets["instances"], start=1):
                img = build_direct_display(p)
                if img is not None:
                    imgs.append(img)
                    caps.append(f"{leaf} #{i}")

            if imgs:
                st.image(imgs, caption=caps, width=320)
            else:
                st.info("인스턴스 마스크 이미지를 읽지 못했습니다.")
        else:
            st.info("인스턴스 마스크가 없습니다.")

# Rendering: detail

def render_question_block(
    image_id: str,
    mode: str,
    questions: List[Dict[str, Any]],
    title: str,
    node_id: Optional[str] = None,
) -> None:
    bucket = get_answers_bucket(image_id, mode, node_id)
    answers = bucket.setdefault("answers", {})

    page_state_key = f"page__{safe_token(image_id)}__{safe_token(mode)}__{safe_token(node_id or 'tree')}"
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 0

    total_pages = max(1, math.ceil(len(questions) / QUESTION_PAGE_SIZE))
    current_page = max(0, min(st.session_state[page_state_key], total_pages - 1))
    st.session_state[page_state_key] = current_page

    question_pages = chunked(questions, QUESTION_PAGE_SIZE)
    visible_questions = question_pages[current_page]

    st.markdown(f"#### {title}")
    # st.caption(f"질문 페이지 {current_page + 1} / {total_pages}")

    def render_single_question(q: Dict[str, Any]) -> None:
        widget_key = question_widget_key(image_id, mode, q["id"], node_id=node_id)
        init_widget_value(widget_key, q, answers.get(q["id"]))

        st.markdown(f"**{q['label']}**")

        if q["type"] == "single_choice":
            st.radio(
                label=q["label"],
                options=q["options"],
                key=widget_key,
                horizontal=True,
                label_visibility="collapsed",
            )

        elif q["type"] == "multi_choice":
            current_values = set(st.session_state.get(widget_key, []) or [])
            cols = st.columns(min(3, max(1, len(q["options"]))))

            selected_values = []
            for i, opt in enumerate(q["options"]):
                check_key = f"{widget_key}__{safe_token(opt)}"
                if check_key not in st.session_state:
                    st.session_state[check_key] = opt in current_values

                with cols[i % len(cols)]:
                    checked = st.checkbox(opt, key=check_key)

                if checked:
                    selected_values.append(opt)

            st.session_state[widget_key] = selected_values

        elif q["type"] == "text":
            st.text_area(
                q["label"],
                key=widget_key,
                height=80,
                label_visibility="collapsed",
            )

    with st.container(height=400):
        triple_buffer = []

        for q in visible_questions:
            triple_buffer.append(q)

            if len(triple_buffer) == 3:
                cols = st.columns(3, gap="small")

                for idx, buffered_q in enumerate(triple_buffer):
                    with cols[idx]:
                        render_single_question(buffered_q)

                triple_buffer.clear()
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # 남은 것 처리
        if triple_buffer:
            cols = st.columns(3, gap="small")

            for idx, buffered_q in enumerate(triple_buffer):
                with cols[idx]:
                    render_single_question(buffered_q)

            for idx in range(len(triple_buffer), 3):
                with cols[idx]:
                    st.empty()

    sync_questions_to_bucket(image_id, mode, visible_questions, node_id=node_id)
    maybe_autosave_to_supabase()

def render_node_detail(record: Optional[Dict[str, Any]]) -> None:
    if record is None:
        st.info("데이터를 찾지 못했습니다.")
        return

    image_id = record["image_id"]
    selected_mode = st.session_state.selected_mode

    # 항상 hierarchy view를 먼저 보여준다
    render_experimental_tree_panel(record)

    if selected_mode == "tree":
        st.subheader("Inspector")
        st.markdown(
            f"<div class='section-card'><b>{image_id}</b><br/>전체 트리 질문 화면입니다.</div>",
            unsafe_allow_html=True,
        )
        render_question_block(
            image_id,
            "tree",
            tree_questions_for(image_id),
            title="전체 트리 질문",
        )
        render_finalize_box(record)
        return

    node_id = st.session_state.selected_node_id

    if (not node_id) or (node_id not in record["nodes"]):
        fallback_node_id = first_reviewable_node_id(record)
        st.session_state.selected_node_id = fallback_node_id
        node_id = fallback_node_id

    if not node_id:
        st.info("노드를 선택하세요.")
        return

    render_asset_panel(record, node_id)
    render_question_block(image_id, "node", node_questions_for(node_id), title="노드 질문", node_id=node_id)
    render_finalize_box(record)


def render_finalize_box(record: Dict[str, Any]) -> None:
    image_id = record["image_id"]
    report = missing_report(image_id, record)

    missing_node_ids = [item["node_id"] for item in report["missing_nodes"]]
    has_tree_missing = bool(report["tree_missing"])

    if not missing_node_ids and not has_tree_missing:
        return

    st.markdown(
        """
        <div style="
            font-size: 30px;
            font-weight: 700;
            margin-top: 12px;
            margin-bottom: 6px;
        ">
            남은 항목
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    pills: List[str] = [
        f"<span class='finalize-pill'>{human_label(node_id)}</span>"
        for node_id in missing_node_ids
    ]

    if has_tree_missing:
        pills.append("<span class='finalize-pill'>전체 트리</span>")

    st.markdown(
        f"""
        <div style="margin-top:0.25rem; display:flex; flex-wrap:wrap; gap:0.3rem;">
            {''.join(pills)}
        </div>
        """,
        unsafe_allow_html=True,
    )



# Main app

def ensure_app_state() -> None:
    defaults = {
        "reviewer_id": "",
        "reviewer_authenticated": False,
        "selected_image_id": None,
        "selected_node_id": None,
        "selected_mode": "node",
        "annotations": {},
        "image_search": "",
        "_loaded_annotation_path": None,
        "_auth_error": "",
        "last_db_save_ts": 0.0,
        "annotations_dirty": False,
        "prefetched_image_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            

def render_reviewer_gate() -> None:
    st.markdown(
        """
        <div style="
            max-width: 520px;
            margin: 400px auto 0 auto;
        ">
            <div style="
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            ">
                Reviewer Login
            </div>
            <div style="
                color: rgba(148,163,184,1);
                font-size: 0.95rem;
                margin-bottom: 1.2rem;
            ">
                Reviewer ID와 비밀번호를 입력하세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([2.6, 1.5, 2.6])
    with center:
        reviewer_input = st.text_input(
            "Reviewer ID",
            key="reviewer_id_input",
            placeholder="아이디를 입력하세요.",
            label_visibility="collapsed",
        )

        password_input = st.text_input(
            "Password",
            key="reviewer_password_input",
            placeholder="비밀번호를 입력하세요.",
            type="password",
            label_visibility="collapsed",
        )

        auth_clicked = st.button("로그인", type="primary", use_container_width=True)

        if auth_clicked:
            reviewer_input = reviewer_input.strip()
            password_input = password_input.strip()

            user_cfg = REVIEW_USERS.get(reviewer_input)

            if not reviewer_input:
                st.session_state.reviewer_authenticated = False
                st.session_state._auth_error = "Reviewer ID를 입력하세요."
            elif not password_input:
                st.session_state.reviewer_authenticated = False
                st.session_state._auth_error = "비밀번호를 입력하세요."
            elif not user_cfg:
                st.session_state.reviewer_authenticated = False
                st.session_state._auth_error = "등록되지 않은 Reviewer ID입니다."
            elif password_input != user_cfg["password"]:
                st.session_state.reviewer_authenticated = False
                st.session_state._auth_error = "비밀번호가 올바르지 않습니다."
            else:
                st.session_state.reviewer_id = reviewer_input
                st.session_state.reviewer_authenticated = True
                st.session_state._auth_error = ""
                st.session_state.selected_image_id = None
                st.session_state.selected_node_id = None
                st.session_state.selected_mode = "node"
                load_annotations_if_needed()
                st.rerun()

        if st.session_state.get("_auth_error"):
            st.error(st.session_state["_auth_error"])

def render_sidebar(records: Dict[str, Any]) -> None:
    with st.sidebar:
        st.header("설정")
        st.write(f"Reviewer ID: **{st.session_state.reviewer_id}**")

        if st.button("로그아웃", use_container_width=True):
            persist_annotations()
            maybe_autosave_to_supabase(force=True)
            st.session_state.reviewer_authenticated = False
            st.session_state.reviewer_id = ""
            st.session_state.selected_image_id = None
            st.session_state.selected_node_id = None
            st.session_state.selected_mode = "node"
            st.session_state.annotations = {}
            st.session_state._loaded_annotation_path = None
            st.rerun()

        load_annotations_if_needed()

        st.markdown("---")
        st.subheader("DB 저장")
        st.caption("약 15초 간격으로 자동 저장됩니다.")

        if st.button("☁️ 클라우드에 저장", use_container_width=True):
            persist_annotations()
            maybe_autosave_to_supabase(force=True)

        st.download_button(
            "현재 annotation JSON 다운로드",
            data=json.dumps(st.session_state.annotations, ensure_ascii=False, indent=2),
            file_name=current_annotation_path().name,
            mime="application/json",
            use_container_width=True,
        )



def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_app_state()
    inject_global_css()

    # 인증 전에는 아무것도 안 보여줌
    if not st.session_state.reviewer_authenticated:
        render_reviewer_gate()
        st.stop()

    # all_records = scan_dataset(DATASET_ROOT)
    # all_records = load_all_records(str(DATASET_ROOT))
    # records = filter_records_by_reviewer(all_records, st.session_state.reviewer_id)
    records = load_records_for_reviewer(st.session_state.reviewer_id)

    render_sidebar(records)
    load_annotations_if_needed()

    # if not all_records:
    #     st.title(APP_TITLE)
    #     st.error(f"내부 dataset root를 찾지 못했어: {DATASET_ROOT}")
    #     st.stop()

    if not records:
        st.title(APP_TITLE)
        st.error(f"Reviewer `{st.session_state.reviewer_id}` 에게 할당된 이미지가 없습니다.")
        st.stop()

    if st.session_state.selected_image_id not in records:
        st.session_state.selected_image_id = next(iter(records.keys()), None)
        if st.session_state.selected_image_id:
            st.session_state.selected_node_id = first_reviewable_node_id(records[st.session_state.selected_image_id])

    selected_record = records.get(st.session_state.selected_image_id)

    if selected_record and st.session_state.selected_mode == "node":
        if st.session_state.selected_node_id not in selected_record["nodes"]:
            actual_nodes = selected_record["actual_nodes"]
            st.session_state.selected_node_id = first_reviewable_node_id(selected_record)


    # if selected_record:
    #     prefetched = st.session_state.get("prefetched_image_ids", [])
    #     image_id = selected_record["image_id"]

    #     if image_id not in prefetched:
    #         prefetch_image_assets(selected_record, include_instances=False)
    #         prefetched.append(image_id)
    #         st.session_state["prefetched_image_ids"] = prefetched


    # col_left, col_mid, col_right = st.columns([0.5, 0.5, 3])
    col_left, col_right = st.columns([0.5, 3])
    with col_left:
        render_image_list(records)
    # with col_mid:
    #     render_tree_panel(selected_record)
    with col_right:
        render_node_detail(selected_record)

if __name__ == "__main__":
    main()