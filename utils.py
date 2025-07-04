# utils.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import subprocess, shutil, re

import json, pandas as pd
import cairosvg                # fallback if pdf2svg CLI is absent
# convert_from_path is still imported for backward compatibility (PNG helper)
from pdf2image import convert_from_path
from collections import defaultdict


# ──────────────────────────────────────────── global paths & experiment map ──
ROOT_DATA = Path("../LLM_MARL/data")

def _discover_experiments(root: Path = ROOT_DATA) -> dict[str, Path]:
    """
    Scan recursively under ROOT_DATA and collect every folder that contains
    a summary.csv.  Return {"relative/label": Path_to_folder, …}.
    """
    exps: dict[str, Path] = {}
    for csv in root.glob("**/summary.csv"):
        label = str(csv.parent.relative_to(root))     # e.g. "o3-mini/o3-mini-1/seed0"
        exps[label] = csv.parent
    return dict(sorted(exps.items()))                 # alphabetical for convenience

EXPERIMENTS: dict[str, Path] = _discover_experiments()
DEFAULT_LABEL = next(iter(EXPERIMENTS)) if EXPERIMENTS else ""


# ───────────────────────────────────────────────────────────── Loader ────
@lru_cache(maxsize=None)
def load_game(
    label: str | None = None,
    source: str = "jsonl",              # "jsonl"  (default)  or "csv"
) -> pd.DataFrame:
    """
    Load one experiment into a DataFrame.

    Parameters
    ----------
    label   : experiment key from `EXPERIMENTS`. Defaults to first experiment.
    source  : "jsonl" -> read record.jsonl (preferred, keeps commas/newlines)
              "csv"  -> read summary.csv  (legacy)
    """
    label = label or DEFAULT_LABEL
    if label not in EXPERIMENTS:
        raise ValueError(f"No experiment named '{label}' in {ROOT_DATA!s}")

    exp_dir = EXPERIMENTS[label]

    # ── read the chosen file ──────────────────────────────────────────────
    if source == "jsonl":
        json_path = exp_dir / "record.jsonl"
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)

    elif source == "csv":
        csv_path = exp_dir / "summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
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
def load_requests_by_round(label: str) -> dict[tuple[str, int], list[dict]]:
    """
    {(agent, round_id) : [request, …]}

    • If a log entry has a key `"round"`, that number is used directly.
    • Otherwise we fall back to the old heuristic:
        - first agent to appear in a round is the leader
        - when a different agent appears, leader’s turn ends
        - next time the leader appears ⇒ new round
    Round counting starts at 1.
    """
    path = EXPERIMENTS[label] / "chat_log.jsonl"
    if not path.exists():
        return {}

    table      = defaultdict(list)
    round_i    = 1          # heuristic counter
    leader     = None
    seen_other = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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
def pdf_to_png(label: str, round_id: int) -> Path:
    """
    Convert round_{round_id}.pdf to PNG (cached) for the given experiment.
    Still here in case some part of the app prefers rasters.
    """
    render_dir = EXPERIMENTS[label] / "renders"
    pdf_path   = render_dir / f"round_{round_id}.pdf"
    png_path   = render_dir / f"round_{round_id}.png"

    if not png_path.exists():
        pages = convert_from_path(pdf_path, dpi=120)
        pages[0].save(png_path)                       # one-page PDF
    return png_path


# ───────────────────────────────────────────────────── PDF ▶ SVG (preferred) ─
@lru_cache(maxsize=None)
def pdf_to_svg(label: str, round_id: int) -> Path:
    """
    Convert round_{round_id}.pdf to SVG (cached) for the given experiment,
    keeping the map fully vector.
    """
    render_dir = EXPERIMENTS[label] / "renders"
    pdf_path   = render_dir / f"round_{round_id}.pdf"
    svg_path   = render_dir / f"round_{round_id}.svg"

    if svg_path.exists():
        return svg_path

    # (a) Try the pdf2svg CLI if available
    if shutil.which("pdf2svg"):
        subprocess.run(["pdf2svg", str(pdf_path), str(svg_path)], check=True)
    else:
        # (b) Pure-Python fallback
        cairosvg.svg_from_pdf(url=str(pdf_path), write_to=str(svg_path))

    return svg_path


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
