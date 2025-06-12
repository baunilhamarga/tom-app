# utils.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import subprocess, shutil, re

import pandas as pd
import cairosvg                # fallback if pdf2svg CLI is absent
# convert_from_path is still imported for backward compatibility (PNG helper)
from pdf2image import convert_from_path


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


# ───────────────────────────────────────────────────────────── CSV loader ────
@lru_cache(maxsize=None)
def load_game(label: str | None = None) -> pd.DataFrame:
    """
    Load summary.csv for the chosen experiment *label*.
    If label is None, fall back to the first discovered experiment.
    """
    label = label or DEFAULT_LABEL
    if label not in EXPERIMENTS:
        raise ValueError(f"No experiment named '{label}' in {ROOT_DATA!s}")

    csv_path = EXPERIMENTS[label] / "summary.csv"
    df = pd.read_csv(csv_path)

    # drop stray "Unnamed: x" columns and sort deterministically
    df = (
        df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
          .sort_values(["round", "agent_id"])
          .reset_index(drop=True)
    )
    return df


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
