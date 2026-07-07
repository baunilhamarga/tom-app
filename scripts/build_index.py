#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_store import _env, build_index_records, write_index_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the dashboard experiment metadata index.")
    parser.add_argument("--data-root", default=None, help="Local experiment root to scan. Defaults to TOM_APP_DATA_ROOT or data.")
    parser.add_argument("--output-dir", default="metadata", help="Where to write runs.jsonl and runs.csv.")
    parser.add_argument(
        "--artifact-uri-prefix",
        default=None,
        help="Optional artifact prefix, e.g. gs://bucket/tom-app/experiments.",
    )
    args = parser.parse_args()

    data_root = args.data_root or _env("TOM_APP_DATA_ROOT", "LLM_MARL_DATA_ROOT", default="data")
    records = build_index_records(data_root, artifact_uri_prefix=args.artifact_uri_prefix)
    jsonl_path, csv_path = write_index_records(records, args.output_dir)
    print(f"Wrote {len(records)} runs to {jsonl_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
