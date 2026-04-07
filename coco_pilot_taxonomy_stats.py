
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def is_image_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(100.0, float(q)))
    idx = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def safe_median(xs: List[float]) -> float:
    return float(median(xs)) if xs else 0.0


def semantic_path_from_node_payload(node_payload: Dict[str, Any]) -> List[str]:
    if isinstance(node_payload.get("semantic_path"), list):
        return [str(x) for x in node_payload["semantic_path"] if str(x).strip()]
    if isinstance(node_payload.get("path"), list):
        path = [str(x) for x in node_payload["path"] if str(x).strip()]
        if len(path) >= 2 and path[1] == "others":
            return [path[0]] + path[2:]
        return path
    label = str(node_payload.get("label", "")).strip()
    return ["root", label] if label else ["root"]


def analyze_tree_json(
    image_dir: Path,
    *,
    exclude_others: bool = False,
) -> Optional[Dict[str, Any]]:
    tree_path = image_dir / "tree.json"
    if not tree_path.exists():
        return None

    tree = load_json(tree_path)
    nodes = tree.get("nodes", {}) or {}
    if not isinstance(nodes, dict):
        return None

    kept_nodes: Dict[str, Dict[str, Any]] = {}
    for node_id, payload in nodes.items():
        if not isinstance(payload, dict):
            continue
        label = str(payload.get("label", "")).strip()
        if not label:
            continue
        if exclude_others and label.lower() == "others":
            continue
        kept_nodes[str(node_id)] = payload

    if not kept_nodes:
        return {
            "image_id": image_dir.name,
            "node_count": 0,
            "leaf_count": 0,
            "root_count": 0,
            "max_semantic_depth": 0,
            "labels": [],
            "semantic_paths": [],
            "edges": [],
            "branching_factors": [],
            "taxonomy_units": [],
        }

    semantic_paths: List[Tuple[str, ...]] = []
    edges: List[Tuple[str, str]] = []
    taxonomy_units: List[str] = []
    branching_factors: List[int] = []
    root_count = 0
    leaf_count = 0
    max_semantic_depth = 0

    kept_node_ids = set(kept_nodes.keys())

    for node_id, payload in kept_nodes.items():
        label = str(payload.get("label", "")).strip()
        semantic_path = semantic_path_from_node_payload(payload)
        semantic_paths.append(tuple(semantic_path))
        taxonomy_units.append("/".join(semantic_path))
        sem_depth = int(payload.get("semantic_depth", max(0, len(semantic_path) - 1)))
        max_semantic_depth = max(max_semantic_depth, sem_depth)

        parent_id = payload.get("semantic_parent_id", payload.get("parent_id"))
        if parent_id is None or str(parent_id) not in kept_node_ids:
            root_count += 1
        else:
            parent_payload = kept_nodes.get(str(parent_id), {})
            parent_label = str(parent_payload.get("label", "")).strip()
            if parent_label:
                edges.append((parent_label, label))

        child_ids = payload.get("children", []) or []
        kept_children = [cid for cid in child_ids if str(cid) in kept_node_ids]
        if len(kept_children) == 0:
            leaf_count += 1
        branching_factors.append(len(kept_children))

    return {
        "image_id": image_dir.name,
        "node_count": len(kept_nodes),
        "leaf_count": leaf_count,
        "root_count": root_count,
        "max_semantic_depth": max_semantic_depth,
        "labels": [str(payload.get("label", "")).strip() for payload in kept_nodes.values()],
        "semantic_paths": semantic_paths,
        "edges": edges,
        "branching_factors": branching_factors,
        "taxonomy_units": taxonomy_units,
    }


def analyze_dataset(
    dataset_root: Path,
    *,
    max_images: Optional[int] = None,
    exclude_others: bool = False,
) -> Dict[str, Any]:
    image_dirs = sorted([p for p in dataset_root.iterdir() if is_image_dir(p)])
    if max_images is not None and max_images > 0:
        image_dirs = image_dirs[:max_images]

    image_summaries: List[Dict[str, Any]] = []
    total_nodes = 0
    total_leaves = 0
    total_roots = 0

    nodes_per_image: List[int] = []
    leaves_per_image: List[int] = []
    roots_per_image: List[int] = []
    max_depths: List[int] = []
    branching_all: List[int] = []

    label_counter: Counter[str] = Counter()
    semantic_path_counter: Counter[str] = Counter()
    edge_counter: Counter[Tuple[str, str]] = Counter()

    for image_dir in image_dirs:
        item = analyze_tree_json(image_dir, exclude_others=exclude_others)
        if item is None:
            continue

        image_summaries.append(item)

        total_nodes += int(item["node_count"])
        total_leaves += int(item["leaf_count"])
        total_roots += int(item["root_count"])

        nodes_per_image.append(int(item["node_count"]))
        leaves_per_image.append(int(item["leaf_count"]))
        roots_per_image.append(int(item["root_count"]))
        max_depths.append(int(item["max_semantic_depth"]))
        branching_all.extend(int(x) for x in item["branching_factors"])

        for label in item["labels"]:
            label_counter[label] += 1
        for path in item["semantic_paths"]:
            semantic_path_counter["/".join(path)] += 1
        for edge in item["edges"]:
            edge_counter[edge] += 1

    sorted_nodes_per_image = sorted(nodes_per_image)
    sorted_depths = sorted(max_depths)
    sorted_branching = sorted(branching_all)

    return {
        "dataset_root": str(dataset_root),
        "num_images": len(image_summaries),
        "exclude_others": bool(exclude_others),
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
        "total_roots": total_roots,
        "unique_labels": len(label_counter),
        "unique_semantic_paths": len(semantic_path_counter),
        "unique_taxonomy_edges": len(edge_counter),
        "avg_nodes_per_image": safe_mean(nodes_per_image),
        "median_nodes_per_image": safe_median(nodes_per_image),
        "p95_nodes_per_image": percentile(sorted_nodes_per_image, 95),
        "avg_leaves_per_image": safe_mean(leaves_per_image),
        "avg_roots_per_image": safe_mean(roots_per_image),
        "avg_max_depth_per_image": safe_mean(max_depths),
        "max_depth_observed": max(max_depths) if max_depths else 0,
        "avg_branching_factor": safe_mean(branching_all),
        "median_branching_factor": safe_median(branching_all),
        "p95_branching_factor": percentile(sorted_branching, 95),
        "p95_max_depth_per_image": percentile(sorted_depths, 95),
        "top_30_labels": label_counter.most_common(30),
        "top_30_semantic_paths": semantic_path_counter.most_common(30),
        "top_30_taxonomy_edges": [
            {"parent": parent, "child": child, "count": count}
            for (parent, child), count in edge_counter.most_common(30)
        ],
        "per_image": [
            {
                "image_id": x["image_id"],
                "node_count": x["node_count"],
                "leaf_count": x["leaf_count"],
                "root_count": x["root_count"],
                "max_semantic_depth": x["max_semantic_depth"],
            }
            for x in image_summaries
        ],
    }


def print_summary(stats: Dict[str, Any]) -> None:
    print("\n===== COCO PILOT TAXONOMY STATS =====")
    print(f"Dataset root: {stats['dataset_root']}")
    print(f"Images: {stats['num_images']}")
    print(f"Exclude 'others': {stats['exclude_others']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total leaves: {stats['total_leaves']}")
    print(f"Total roots: {stats['total_roots']}")
    print(f"Unique labels (word types): {stats['unique_labels']}")
    print(f"Unique semantic paths (taxonomy units): {stats['unique_semantic_paths']}")
    print(f"Unique parent->child edge types: {stats['unique_taxonomy_edges']}")
    print(f"Avg nodes/image: {stats['avg_nodes_per_image']:.2f}")
    print(f"Median nodes/image: {stats['median_nodes_per_image']:.2f}")
    print(f"P95 nodes/image: {stats['p95_nodes_per_image']:.2f}")
    print(f"Avg leaves/image: {stats['avg_leaves_per_image']:.2f}")
    print(f"Avg roots/image: {stats['avg_roots_per_image']:.2f}")
    print(f"Avg max depth/image: {stats['avg_max_depth_per_image']:.2f}")
    print(f"Max depth observed: {stats['max_depth_observed']}")
    print(f"Avg branching factor: {stats['avg_branching_factor']:.2f}")
    print(f"Median branching factor: {stats['median_branching_factor']:.2f}")
    print(f"P95 branching factor: {stats['p95_branching_factor']:.2f}")

    print("\n--- Top 20 labels ---")
    for label, count in stats["top_30_labels"][:20]:
        print(f"{label}: {count}")

    print("\n--- Top 20 semantic paths ---")
    for path, count in stats["top_30_semantic_paths"][:20]:
        print(f"{path}: {count}")

    print("\n--- Top 20 taxonomy edges ---")
    for row in stats["top_30_taxonomy_edges"][:20]:
        print(f"{row['parent']} -> {row['child']}: {row['count']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--exclude_others", action="store_true")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    stats = analyze_dataset(
        dataset_root=dataset_root,
        max_images=None if int(args.max_images) <= 0 else int(args.max_images),
        exclude_others=bool(args.exclude_others),
    )
    print_summary(stats)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
