from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATASET_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1")
MAX_IMAGES = 1000

SAVE_DIR = Path("./analysis_results")
SAVE_DIR.mkdir(exist_ok=True)


def is_image_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def is_node_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("root__")


def get_depth(node_name: str) -> int:
    return node_name.count("__")


def count_instances(node_dir: Path) -> int:
    """
    노드 폴더 안의 instance 수를 센다.
    우선 *.mask.png 개수를 사용.
    만약 하나도 없으면 0으로 둔다.
    """
    mask_files = list(node_dir.glob("*.mask.png"))
    return len(mask_files)


def parent_node_name(node_name: str) -> str | None:
    """
    root__pavilion__roof -> root__pavilion
    root__pavilion -> None
    """
    parts = node_name.split("__")
    if len(parts) <= 2:
        return None
    return "__".join(parts[:-1])


def compute_branching_factors_from_dirs(node_names: list[str]) -> list[int]:
    """
    폴더명만으로 트리의 branching factor를 계산한다.
    각 노드가 direct child를 몇 개 가지는지 센다.
    """
    children_map = defaultdict(list)
    node_set = set(node_names)

    for node in node_names:
        parent = parent_node_name(node)
        if parent is not None and parent in node_set:
            children_map[parent].append(node)

    branching_factors = [len(children) for children in children_map.values()]
    return branching_factors


def analyze():
    image_dirs = sorted([p for p in DATASET_ROOT.iterdir() if is_image_dir(p)])[:MAX_IMAGES]

    total_nodes = 0
    total_instances = 0

    nodes_per_image = []
    instances_per_node = []
    instances_per_image = []

    depth_counter = defaultdict(int)
    branching_factors = []

    for img_dir in image_dirs:
        node_dirs = sorted([p for p in img_dir.iterdir() if is_node_dir(p)])
        node_names = [p.name for p in node_dirs]

        num_nodes = len(node_dirs)
        total_nodes += num_nodes
        nodes_per_image.append(num_nodes)

        image_instance_sum = 0

        for nd in node_dirs:
            depth = get_depth(nd.name)
            depth_counter[depth] += 1

            inst = count_instances(nd)
            instances_per_node.append(inst)
            image_instance_sum += inst

        total_instances += image_instance_sum
        instances_per_image.append(image_instance_sum)

        branching_factors.extend(compute_branching_factors_from_dirs(node_names))

    return {
        "num_images": len(image_dirs),
        "total_nodes": total_nodes,
        "total_instances": total_instances,
        "nodes_per_image": nodes_per_image,
        "instances_per_node": instances_per_node,
        "instances_per_image": instances_per_image,
        "depth_counter": depth_counter,
        "branching_factors": branching_factors,
    }


def plot_hist(data, title, filename, bins=30):
    if not data:
        return

    data = np.array(data)
    weights = np.ones_like(data) / len(data) * 100  # 퍼센트 변환

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, weights=weights)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=200)
    plt.close()


def plot_depth(depth_counter):
    if not depth_counter:
        return

    depths = sorted(depth_counter.keys())
    counts = np.array([depth_counter[d] for d in depths])
    total = counts.sum()

    percentages = counts / total * 100

    plt.figure(figsize=(6, 4))
    plt.bar(depths, percentages)

    plt.title("Depth Distribution")
    plt.xlabel("Depth")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "depth_distribution.png", dpi=200)
    plt.close()

def plot_cdf(data, title, filename, percentile_line=95):
    if not data:
        return

    data = np.sort(np.array(data))
    cdf = np.arange(1, len(data) + 1) / len(data) * 100

    threshold_value = np.percentile(data, percentile_line)

    plt.figure(figsize=(6, 4))
    plt.plot(data, cdf)

    plt.axhline(y=percentile_line, linestyle='--')

    plt.axvline(x=threshold_value, linestyle='--')

    plt.scatter([threshold_value], [percentile_line])

    plt.text(
        threshold_value,
        percentile_line,
        f"  {percentile_line}% → {threshold_value:.1f}",
        verticalalignment='bottom'
    )

    plt.xlabel("Value")
    plt.ylabel("Cumulative %")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=200)
    plt.close()

    return threshold_value

def main():
    stats = analyze()

    num_images = stats["num_images"]
    total_nodes = stats["total_nodes"]
    total_instances = stats["total_instances"]

    avg_nodes_per_image = total_nodes / num_images if num_images else 0.0
    avg_instances_per_image = total_instances / num_images if num_images else 0.0
    avg_instances_per_node = total_instances / total_nodes if total_nodes else 0.0

    print("\n===== DATASET STATS (first 1000 images) =====")
    print(f"Images: {num_images}")
    print(f"Nodes: {total_nodes}")
    print(f"Instances: {total_instances}")
    print(f"Avg nodes per image: {avg_nodes_per_image:.2f}")
    print(f"Avg instances per image: {avg_instances_per_image:.2f}")
    print(f"Avg instances per node: {avg_instances_per_node:.2f}")

    summary_lines = [
        "===== DATASET STATS (first 1000 images) =====",
        f"Images: {num_images}",
        f"Nodes: {total_nodes}",
        f"Instances: {total_instances}",
        f"Avg nodes per image: {avg_nodes_per_image:.2f}",
        f"Avg instances per image: {avg_instances_per_image:.2f}",
        f"Avg instances per node: {avg_instances_per_node:.2f}",
    ]
    (SAVE_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    plot_hist(stats["nodes_per_image"], "Nodes per Image", "nodes_per_image.png")
    plot_hist(stats["instances_per_node"], "Instances per Node", "instances_per_node.png")
    plot_hist(stats["instances_per_image"], "Instances per Image", "instances_per_image.png")
    plot_hist(stats["branching_factors"], "Branching Factor", "branching_factor.png")
    plot_depth(stats["depth_counter"])

    p1 = plot_cdf(stats["nodes_per_image"], "CDF: Nodes per Image", "cdf_nodes.png")
    p2 = plot_cdf(stats["instances_per_node"], "CDF: Instances per Node", "cdf_instances_node.png")
    p3 = plot_cdf(stats["branching_factors"], "CDF: Branching Factor", "cdf_branching.png")

    print(f"95% nodes/image <= {p1:.2f}")
    print(f"95% instances/node <= {p2:.2f}")
    print(f"95% branching factor <= {p3:.2f}")

if __name__ == "__main__":
    main()