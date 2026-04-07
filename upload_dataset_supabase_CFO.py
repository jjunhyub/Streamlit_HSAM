from __future__ import annotations

import csv
import mimetypes
import time
from pathlib import Path
from typing import Iterator

from supabase import create_client

# 직접 수정할 값들
SUPABASE_URL = "https://vutkowvxahuggkmhrtka.supabase.co"
SUPABASE_KEY = ""
BUCKET_NAME = "review-dataset"

# 네 로컬 데이터셋 폴더
LOCAL_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1").resolve()

# Storage 안에서 시작될 폴더명
REMOTE_PREFIX = "coco_pilot_v1"

# 이번 스크립트는 visual asset만 강제 갱신하는 목적이므로 기본값을 upsert=True로 둔다.
UPSERT = True

# 재시도 횟수
MAX_RETRIES = 3

# 업로드 대상 suffix
TARGET_SUFFIXES = (
    ".instances.colored.png",
    ".mask.original.full.png",
    ".overlay.png",
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def iter_target_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(TARGET_SUFFIXES):
            yield path


def guess_content_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def to_remote_key(local_path: Path) -> str:
    rel = local_path.relative_to(LOCAL_ROOT).as_posix()
    return f"{REMOTE_PREFIX}/{rel}"


def upload_one(local_path: Path) -> tuple[str, str, int]:
    remote_key = to_remote_key(local_path)
    content_type = guess_content_type(local_path)
    size_bytes = local_path.stat().st_size

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(local_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=remote_key,
                    file=f,
                    file_options={
                        "cache-control": "3600",
                        "upsert": "true" if UPSERT else "false",
                        "content-type": content_type,
                    },
                )
            return remote_key, content_type, size_bytes

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Upload failed for {remote_key}: {e}") from e

            wait_sec = 2 ** (attempt - 1)
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {remote_key} -> {e}")
            time.sleep(wait_sec)

    raise RuntimeError(f"Upload failed: {remote_key}")


def asset_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".instances.colored.png"):
        return "colored"
    if name.endswith(".mask.original.full.png"):
        return "mask_full"
    if name.endswith(".overlay.png"):
        return "overlay"
    return "other"


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"LOCAL_ROOT not found: {LOCAL_ROOT}")

    files = list(iter_target_files(LOCAL_ROOT))
    total = len(files)
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"LOCAL_ROOT: {LOCAL_ROOT}")
    print(f"BUCKET_NAME: {BUCKET_NAME}")
    print(f"REMOTE_PREFIX: {REMOTE_PREFIX}")
    print(f"UPSERT: {UPSERT}")
    print(f"Target suffixes: {', '.join(TARGET_SUFFIXES)}")
    print(f"Found visual asset files: {total}")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")

    manifest_path = Path("upload_visual_assets_manifest.csv")
    error_log_path = Path("upload_visual_assets_errors.csv")

    manifest_exists = manifest_path.exists()
    error_exists = error_log_path.exists()

    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_f, \
         error_log_path.open("a", newline="", encoding="utf-8") as error_f:

        manifest_writer = csv.writer(manifest_f)
        error_writer = csv.writer(error_f)

        if not manifest_exists:
            manifest_writer.writerow([
                "local_path",
                "remote_key",
                "asset_kind",
                "content_type",
                "size_bytes",
            ])

        if not error_exists:
            error_writer.writerow(["local_path", "remote_key", "asset_kind", "error"])

        done = 0
        failed = 0
        kind_counts = {"colored": 0, "mask_full": 0, "overlay": 0}

        for idx, local_path in enumerate(files, start=1):
            kind = asset_kind(local_path)
            remote_key = to_remote_key(local_path)

            try:
                remote_key, content_type, size_bytes = upload_one(local_path)
                manifest_writer.writerow([
                    str(local_path),
                    remote_key,
                    kind,
                    content_type,
                    size_bytes,
                ])
                manifest_f.flush()
                done += 1
                if kind in kind_counts:
                    kind_counts[kind] += 1

            except Exception as e:
                error_writer.writerow([str(local_path), remote_key, kind, str(e)])
                error_f.flush()
                failed += 1
                print(f"[FAIL] {remote_key} -> {e}")

            if idx % 50 == 0 or idx == total:
                print(
                    f"[{idx}/{total}] done={done}, failed={failed}, "
                    f"colored={kind_counts['colored']}, mask_full={kind_counts['mask_full']}, overlay={kind_counts['overlay']}"
                )

    print("\nDone.")
    print(f"Manifest saved to: {manifest_path.resolve()}")
    print(f"Errors saved to: {error_log_path.resolve()}")


if __name__ == "__main__":
    main()
