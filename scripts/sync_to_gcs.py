#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_store import (
    _env,
    _gcs_client,
    build_index_records,
    upload_file_to_gcs,
    write_index_records,
)


def _blob_name(prefix: str, *parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part.strip("/")]
    if prefix.strip("/"):
        clean_parts.insert(0, prefix.strip("/"))
    return "/".join(clean_parts)


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _generate_missing_svgs(root: Path) -> int:
    converter = None
    if shutil.which("pdf2svg"):
        converter = "pdf2svg"
    elif shutil.which("pdftocairo"):
        converter = "pdftocairo"
    if converter is None:
        print("No pdf2svg or pdftocairo found; uploading existing render files only.")
        return 0

    generated = 0
    for pdf_path in sorted(root.glob("**/renders/round_*.pdf")):
        svg_path = pdf_path.with_suffix(".svg")
        if svg_path.exists():
            continue
        if converter == "pdf2svg":
            subprocess.run(["pdf2svg", str(pdf_path), str(svg_path)], check=True)
        else:
            subprocess.run(["pdftocairo", "-svg", str(pdf_path), str(svg_path)], check=True)
        generated += 1
    return generated


def _count_missing_svgs(root: Path) -> int:
    return sum(1 for pdf_path in root.glob("**/renders/round_*.pdf") if not pdf_path.with_suffix(".svg").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload dashboard experiments and metadata to Google Cloud Storage.")
    parser.add_argument("--bucket", default=None, help="GCS bucket name. Defaults to GCS_BUCKET_NAME/TOM_APP_GCS_BUCKET.")
    parser.add_argument("--prefix", default=None, help="GCS prefix inside the bucket. Defaults to TOM_APP_GCS_PREFIX or tom-app.")
    parser.add_argument("--data-root", default=None, help="Local experiment root to upload. Defaults to TOM_APP_DATA_ROOT or data.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading.")
    parser.add_argument("--metadata-only", action="store_true", help="Only upload metadata/runs.*.")
    parser.add_argument("--no-generate-svgs", action="store_true", help="Do not create missing render SVGs before upload.")
    args = parser.parse_args()
    args.bucket = args.bucket or _env("TOM_APP_GCS_BUCKET", "GCS_BUCKET_NAME")
    args.prefix = args.prefix if args.prefix is not None else (_env("TOM_APP_GCS_PREFIX", default="tom-app") or "tom-app")
    if not args.bucket:
        parser.error("--bucket is required unless GCS_BUCKET_NAME or TOM_APP_GCS_BUCKET is set")

    data_root = Path(args.data_root or _env("TOM_APP_DATA_ROOT", "LLM_MARL_DATA_ROOT", default="data"))
    if not args.no_generate_svgs and args.dry_run:
        missing_svgs = _count_missing_svgs(data_root)
        if missing_svgs:
            print(f"DRY RUN would generate {missing_svgs} missing SVG renders before upload.")
    elif not args.no_generate_svgs:
        generated = _generate_missing_svgs(data_root)
        if generated:
            print(f"Generated {generated} missing SVG renders.")

    artifact_prefix = f"gs://{args.bucket}/{_blob_name(args.prefix, 'experiments')}"
    records = build_index_records(data_root, artifact_uri_prefix=artifact_prefix)
    print(f"Indexed {len(records)} runs from {data_root}")

    client = None if args.dry_run else _gcs_client()

    if not args.metadata_only:
        uploaded = 0
        for local_path in _iter_files(data_root):
            rel = local_path.relative_to(data_root).as_posix()
            blob = _blob_name(args.prefix, "experiments", rel)
            if args.dry_run:
                print(f"DRY RUN upload {local_path} -> gs://{args.bucket}/{blob}")
            else:
                upload_file_to_gcs(local_path, args.bucket, blob, client=client)
            uploaded += 1
        print(f"{'Would upload' if args.dry_run else 'Uploaded'} {uploaded} artifact files.")

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path, csv_path = write_index_records(records, tmp)
        for local_path in (jsonl_path, csv_path):
            blob = _blob_name(args.prefix, "metadata", local_path.name)
            if args.dry_run:
                print(f"DRY RUN upload {local_path} -> gs://{args.bucket}/{blob}")
            else:
                upload_file_to_gcs(local_path, args.bucket, blob, client=client)

    print(f"Dashboard index: gs://{args.bucket}/{_blob_name(args.prefix, 'metadata/runs.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
