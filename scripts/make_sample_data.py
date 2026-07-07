#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_store import _env, RENDER_SUFFIXES, RUN_FILES, build_index_records, write_index_records


def _round_sort_key(path: Path) -> tuple[int, str]:
    import re

    match = re.search(r"round_(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _pick_evenly(paths: list[Path], limit: int) -> list[Path]:
    if limit < 0 or len(paths) <= limit:
        return paths
    if limit == 0:
        return []
    if limit == 1:
        return [paths[0]]
    indexes = sorted({round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)})
    return [paths[i] for i in indexes]


def _copy_run(source_root: Path, dest_root: Path, label: str, max_renders: int) -> None:
    source_dir = source_root / label
    dest_dir = dest_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in RUN_FILES:
        source_path = source_dir / filename
        if source_path.exists():
            target = dest_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

    source_renders = source_dir / "renders"
    if not source_renders.exists():
        return
    render_files = [
        path
        for path in sorted(source_renders.iterdir(), key=_round_sort_key)
        if path.is_file() and path.suffix.lower() in RENDER_SUFFIXES
    ]
    for source_path in _pick_evenly(render_files, max_renders):
        target = dest_dir / "renders" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a compact sample_data directory for the dashboard repo.")
    parser.add_argument("--source", default=None, help="Full local data root. Defaults to TOM_APP_DATA_ROOT or data.")
    parser.add_argument("--dest", default="sample_data", help="Sample output root.")
    parser.add_argument("--models", type=int, default=3, help="Maximum distinct models to include.")
    parser.add_argument("--seeds", type=int, default=3, help="Maximum seeds per model/experiment.")
    parser.add_argument(
        "--max-renders-per-run",
        type=int,
        default=4,
        help="Maximum render files per run; use -1 to copy all renders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination directory first.")
    args = parser.parse_args()

    source_root = Path(args.source or _env("TOM_APP_DATA_ROOT", "LLM_MARL_DATA_ROOT", default="data"))
    dest_root = Path(args.dest)
    if args.overwrite and dest_root.exists():
        shutil.rmtree(dest_root)

    records = build_index_records(source_root)
    model_order = []
    for record in records:
        model = str(record.get("model") or record["label"].split("/")[0])
        if model not in model_order:
            model_order.append(model)
    selected_models = set(model_order[: args.models])

    seeds_by_exp: dict[tuple[str, str], set[str]] = defaultdict(set)
    selected = []
    for record in records:
        model = str(record.get("model") or record["label"].split("/")[0])
        experiment = str(record.get("experiment") or record["label"].split("/")[1])
        seed = str(record.get("seed") or record["label"].split("/")[-1])
        key = (model, experiment)
        if model not in selected_models:
            continue
        if seed not in seeds_by_exp[key] and len(seeds_by_exp[key]) >= args.seeds:
            continue
        seeds_by_exp[key].add(seed)
        selected.append(record)

    for record in selected:
        _copy_run(source_root, dest_root, record["label"], args.max_renders_per_run)

    sample_records = build_index_records(dest_root)
    write_index_records(sample_records, dest_root / "metadata")
    print(f"Copied {len(sample_records)} sample runs to {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
