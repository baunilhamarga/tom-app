# utils.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import subprocess, shutil, re
import base64
from io import BytesIO

import json, pandas as pd
import cairosvg                # fallback if pdf2svg CLI is absent
# convert_from_path is still imported for backward compatibility (PNG helper)
from pdf2image import convert_from_path
from collections import defaultdict

from experiment_store import ExperimentRef, ExperimentStore, resolve_mission_outcome

# ──────────────────────────────────────────── global paths & experiment map ──
ROOT_DATA = Path("./data")

@lru_cache(maxsize=2)
def get_store(source: str = "full") -> ExperimentStore:
    return ExperimentStore.from_source(source)


def _clear_artifact_caches() -> None:
    load_game.cache_clear()
    load_requests_by_round.cache_clear()
    pdf_to_png.cache_clear()
    pdf_to_svg.cache_clear()


def refresh_store(source: str = "full") -> ExperimentStore:
    get_store.cache_clear()
    _clear_artifact_caches()
    return get_store(source)


def _discover_experiments(root: Path = ROOT_DATA) -> dict[str, ExperimentRef]:
    """
    Discover experiments from Cloud Storage when configured, otherwise from
    sample_data/ or data/. The root argument is kept for backward
    compatibility with older app code.
    """
    global STORE, EXPERIMENTS, DEFAULT_LABEL
    get_store.cache_clear()
    STORE = get_store("full")
    EXPERIMENTS = STORE.experiments
    DEFAULT_LABEL = next(iter(EXPERIMENTS)) if EXPERIMENTS else ""
    return EXPERIMENTS

STORE = get_store("full")
EXPERIMENTS: dict[str, ExperimentRef] = STORE.experiments
DEFAULT_LABEL = next(iter(EXPERIMENTS)) if EXPERIMENTS else ""


def refresh_experiments() -> dict[str, ExperimentRef]:
    _clear_artifact_caches()
    return _discover_experiments()


def get_experiment_dir(label: str, store: ExperimentStore | None = None) -> Path:
    store = store or STORE
    ref = store.get(label)
    if ref.source == "local":
        return Path(ref.artifact_base_uri)
    path = store.cache_dir / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(label: str, relative_path: str, store: ExperimentStore | None = None) -> Path:
    store = store or STORE
    ref = store.get(label)
    if ref.source == "local":
        return Path(ref.artifact_base_uri) / relative_path
    return store.cache_dir / label / relative_path


def artifact_exists(label: str, relative_path: str, store: ExperimentStore | None = None) -> bool:
    return (store or STORE).artifact_exists(label, relative_path)


def load_json_artifact(
    label: str,
    relative_path: str,
    store: ExperimentStore | None = None,
) -> dict:
    try:
        return json.loads((store or STORE).read_text(label, relative_path))
    except Exception:
        return {}


def _loads_concatenated_json(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    idx = 0
    rows = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        rows.append(obj)
    return rows


# ───────────────────────────────────────────────────────────── Loader ────
@lru_cache(maxsize=None)
def load_game(
    label: str | None = None,
    source: str = "jsonl",              # "jsonl"  (default)  or "csv"
    store: ExperimentStore | None = None,
) -> pd.DataFrame:
    """
    Load one experiment into a DataFrame.

    Parameters
    ----------
    label   : experiment key from `EXPERIMENTS`. Defaults to first experiment.
    source  : "jsonl" -> read record.jsonl (preferred, keeps commas/newlines)
              "csv"  -> read summary.csv  (legacy)
    """
    store = store or STORE
    label = label or next(iter(store.experiments), "")
    if label not in store.experiments:
        raise ValueError(f"No experiment named '{label}' in {ROOT_DATA!s}")

    # ── read the chosen file ──────────────────────────────────────────────
    if source == "jsonl":
        try:
            text = store.read_text(label, "record.jsonl")
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except FileNotFoundError:
            text = store.read_text(label, "record.json")
            rows = _loads_concatenated_json(text)
        df = pd.DataFrame(rows)

    elif source == "csv":
        csv_path = store.artifact_path(label, "summary.csv")
        df = pd.read_csv(csv_path)
        # drop stray unnamed columns from the legacy writer
        df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])

    else:
        raise ValueError("source must be 'jsonl' or 'csv'")

    # ── tidy & sort ───────────────────────────────────────────────────────
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype(int)
    df = df.sort_values(["round", "agent_id"]).reset_index(drop=True)
    return df

@lru_cache(maxsize=None)
def load_requests_by_round(
    label: str,
    store: ExperimentStore | None = None,
) -> dict[tuple[str, int], list[dict]]:
    """
    {(agent, round_id) : [request, …]}

    • If a log entry has a key `"round"`, that number is used directly.
    • Otherwise we fall back to the old heuristic:
        - first agent to appear in a round is the leader
        - when a different agent appears, leader’s turn ends
        - next time the leader appears ⇒ new round
    Round counting starts at 1.
    """
    store = store or STORE
    try:
        text = store.read_text(label, "chat_log.jsonl")
    except Exception:
        return {}

    table      = defaultdict(list)
    round_i    = 1          # heuristic counter
    leader     = None
    seen_other = False

    for line in text.splitlines():
        if not line.strip():
            continue
        rec   = json.loads(line)
        agent = rec.get("agent", "unknown")

        # ── use explicit round if present ───────────────────────────
        if "round" in rec:
            try:
                round_num = int(rec["round"])
            except (TypeError, ValueError):
                round_num = round_i           # fallback if malformed
            # synchronise heuristic tracker with explicit value
            round_i    = round_num
            leader     = agent
            seen_other = False


        # ── otherwise apply leader/turn heuristic ───────────────────
        else:
            if leader is None:
                leader = agent
            elif agent == leader and seen_other:
                round_i += 1
                leader, seen_other = agent, False
            elif agent != leader:
                seen_other = True
            round_num = round_i

        table[(agent, round_num)].append(rec)

    return table

# ───────────────────────────────────────────────────── PDF ▶ PNG (optional) ─
@lru_cache(maxsize=None)
def pdf_to_png(
    label: str,
    round_id: int,
    store: ExperimentStore | None = None,
) -> Path:
    """
    Convert round_{round_id}.pdf to PNG (cached) for the given experiment.
    Still here in case some part of the app prefers rasters.
    """
    store = store or STORE
    pdf_path   = store.artifact_path(label, f"renders/round_{round_id}.pdf")
    png_path   = _cache_path(label, f"renders/round_{round_id}.png", store)

    if not png_path.exists():
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pages = convert_from_path(pdf_path, dpi=120)
        pages[0].save(png_path)                       # one-page PDF
    return png_path


# ───────────────────────────────────────────────────── PDF ▶ SVG (preferred) ─
@lru_cache(maxsize=None)
def pdf_to_svg(
    label: str,
    round_id: int,
    store: ExperimentStore | None = None,
) -> Path:
    """
    Convert round_{round_id}.pdf to SVG (cached) for the given experiment,
    keeping the map fully vector.
    """
    store = store or STORE
    ref = store.get(label)
    svg_path = _cache_path(label, f"renders/round_{round_id}.svg", store)

    if svg_path.exists():
        return svg_path

    if ref.source != "local":
        try:
            return store.artifact_path(label, f"renders/round_{round_id}.svg")
        except Exception:
            pass

    pdf_path = store.artifact_path(label, f"renders/round_{round_id}.pdf")
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    _convert_pdf_to_svg(pdf_path, svg_path)
    return svg_path

def pdf_to_svg_file(pdf_path: Path) -> Path:
    """
    Convert <something>.pdf → <something>.svg (once) and return the SVG Path.
    """
    svg_path = pdf_path.with_suffix(".svg")
    if svg_path.exists():
        return svg_path

    _convert_pdf_to_svg(pdf_path, svg_path)
    return svg_path


def _convert_pdf_to_svg(pdf_path: Path, svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdf2svg"):
        subprocess.run(["pdf2svg", str(pdf_path), str(svg_path)], check=True)
    elif shutil.which("pdftocairo"):
        subprocess.run(["pdftocairo", "-svg", str(pdf_path), str(svg_path)], check=True)
    else:
        pages = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        image = pages[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        width, height = image.size
        svg_path.write_text(
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                f'<image width="{width}" height="{height}" '
                f'href="data:image/png;base64,{data}"/></svg>'
            ),
            encoding="utf-8",
        )


# ─────────────────────────────────────────── SVG width patch helper (unchanged)
def expand_svg(svg_xml: str) -> str:
    """
    Inject width=100% & auto height so the SVG stretches across the Streamlit column.
    """
    return re.sub(
        r"<svg\b",
        '<svg style="width:100%;height:auto;"',
        svg_xml,
        count=1,
        flags=re.IGNORECASE,
    )
