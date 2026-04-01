from __future__ import annotations

import csv
import mimetypes
import time
from pathlib import Path

from supabase import create_client

# 직접 수정할 값들
SUPABASE_URL = "https://vutkowvxahuggkmhrtka.supabase.co"
SUPABASE_SECRET_KEY = ""
BUCKET_NAME = "review-dataset"

# 네 로컬 데이터셋 폴더
LOCAL_ROOT = Path(r"C:\Users\junhy\Documents\2026\HSAM\datasets\coco_pilot_v1").resolve()

# Storage 안에서 시작될 폴더명
REMOTE_PREFIX = "coco_pilot_v1"

# 이미 존재하는 파일은 건너뛸지
SKIP_IF_EXISTS = True

# 재시도 횟수
MAX_RETRIES = 3

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def guess_content_type(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".json.gz"):
        return "application/gzip"
    if name.endswith(".json"):
        return "application/json"
    if path.suffix.lower() == ".gz":
        return "application/gzip"

    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def to_remote_key(local_path: Path) -> str:
    rel = local_path.relative_to(LOCAL_ROOT).as_posix()
    return f"{REMOTE_PREFIX}/{rel}"


def is_duplicate_error(msg: str) -> bool:
    msg_lower = msg.lower()
    return (
        "asset already exists" in msg_lower
        or "the resource already exists" in msg_lower
        or "'error': duplicate" in msg_lower
        or '"error": "duplicate"' in msg_lower
        or "statuscode': 409" in msg_lower
        or '"statuscode": 409' in msg_lower
        or ("409" in msg_lower and "exists" in msg_lower)
    )


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
                        "upsert": "true",
                        # "upsert": "false" if SKIP_IF_EXISTS else "true",
                        "content-type": content_type,
                    },
                )
            return remote_key, content_type, size_bytes

        except Exception as e:
            msg = str(e)

            if SKIP_IF_EXISTS and is_duplicate_error(msg):
                print(f"[SKIP] already exists: {remote_key}")
                return remote_key, content_type, size_bytes

            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Upload failed for {remote_key}: {e}") from e

            wait_sec = 2 ** (attempt - 1)
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {remote_key} -> {e}")
            time.sleep(wait_sec)

    raise RuntimeError(f"Upload failed: {remote_key}")

def main():
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"LOCAL_ROOT not found: {LOCAL_ROOT}")

    files = list(iter_files(LOCAL_ROOT))
    total = len(files)
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"LOCAL_ROOT: {LOCAL_ROOT}")
    print(f"BUCKET_NAME: {BUCKET_NAME}")
    print(f"REMOTE_PREFIX: {REMOTE_PREFIX}")
    print(f"Found files: {total}")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")

    manifest_path = Path("upload_manifest.csv")
    error_log_path = Path("upload_errors.csv")

    processed = set()
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["local_path"])

    manifest_exists = manifest_path.exists()
    error_exists = error_log_path.exists()

    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_f, \
         error_log_path.open("a", newline="", encoding="utf-8") as error_f:

        manifest_writer = csv.writer(manifest_f)
        error_writer = csv.writer(error_f)

        if not manifest_exists:
            manifest_writer.writerow(["local_path", "remote_key", "content_type", "size_bytes"])

        if not error_exists:
            error_writer.writerow(["local_path", "remote_key", "error"])

        done = 0
        skipped = 0
        failed = 0

        for idx, local_path in enumerate(files, start=1):
            local_str = str(local_path)

            if local_str in processed:
                skipped += 1
                continue

            try:
                remote_key, content_type, size_bytes = upload_one(local_path)
                manifest_writer.writerow([local_str, remote_key, content_type, size_bytes])
                manifest_f.flush()
                done += 1

            except Exception as e:
                remote_key = to_remote_key(local_path)
                error_writer.writerow([local_str, remote_key, str(e)])
                error_f.flush()
                failed += 1
                print(f"[FAIL] {remote_key} -> {e}")

            if idx % 50 == 0 or idx == total:
                print(f"[{idx}/{total}] done={done}, resumed_skip={skipped}, failed={failed}")

    print("\nDone.")
    print(f"Manifest saved to: {manifest_path.resolve()}")
    print(f"Errors saved to: {error_log_path.resolve()}")
    