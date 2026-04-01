from pathlib import Path
import shutil

root = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1")


clean_file = "clean_strict.txt"

with open(clean_file) as f:
    keep = set(Path(line.strip()).name for line in f if line.strip())

deleted = 0
kept = 0

for d in root.iterdir():
    if not d.is_dir():
        continue

    if d.name in keep:
        kept += 1
    else:
        shutil.rmtree(d)
        deleted += 1

print(f"Kept: {kept}, Deleted: {deleted}")