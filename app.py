"""
Streamlit replay viewer for the EMNLP-23 ToM search-and-rescue game.
Run with:  streamlit run app.py
"""

import time, re, streamlit as st, pandas as pd
from pathlib import Path
import utils                         # we’ll mutate utils.EXPERIMENTS
from utils import load_game, pdf_to_svg, expand_svg
import json
PRICING_PATH = Path(__file__).parent / "assets/pricing.json"

@st.cache_data(show_spinner=False)
def load_pricing() -> list[dict]:
    with open(PRICING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

import re
from typing import Optional, Dict, List

def match_price_row(model_name: str, table: List[Dict]) -> Optional[Dict]:
    """
    Return the *best* pricing entry for a model name.

    Rules
    -----
    1. Strip a final YYYY-MM-DD suffix (e.g.  gpt-4.1-nano-2025-04-14  →  gpt-4.1-nano)
    2. Among all price-table rows whose Model string is **contained** in the
       stripped name, pick the *longest* row['Model']  (i.e. most specific).
    3. If nothing matches, return None.
    """
    # 1) normalise and strip datestamp
    m = model_name.lower()
    m = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)           # remove -YYYY-MM-DD

    best_row = None
    best_len = -1

    for row in table:
        cand = row["Model"].lower()
        if cand in m:
            if len(cand) > best_len:                   # 2) most specific
                best_len = len(cand)
                best_row = row

    return best_row

def estimate_cost(results: dict, price: dict) -> tuple[float, float, float, float]:
    """Return (input$, cached$, output$, total$)."""
    in_tokens   = results.get("prompt_tokens",0) - results.get("prompt_tokens_details.cached_tokens",0)
    cache_tokens= results.get("prompt_tokens_details.cached_tokens",0)
    out_tokens  = results.get("completion_tokens",0)

    input_cost  = in_tokens    / 1_000_000 * price["Input"]
    cached_cost = cache_tokens / 1_000_000 * price["Cached input"]
    output_cost = out_tokens   / 1_000_000 * price["Output"]
    total       = input_cost + cached_cost + output_cost
    return round(input_cost, 4), round(cached_cost, 4), round(output_cost, 4), round(total, 4)


# ── apply scheduled round change (comes from previous run) ────────────────
if "round_pending" in st.session_state:
    st.session_state.round = st.session_state.round_pending
    del st.session_state.round_pending         # consume the flag

# ───────────────────────── helpers ──────────────────────────
def extract_score(obs_text: str) -> int | None:
    m = re.search(r"Total team score:\s*(\d+)", obs_text)
    return int(m.group(1)) if m else None

AGENT_COLORS = {"alpha": "#e6194b", "bravo": "#3cb44b", "charlie": "#4363d8"}
TOM_LABELS   = {
    "ToM1st": "ToM-0th (introspection)",
    "ToM2nd": "ToM-1st (first-order)",
    "ToM3rd": "ToM-2nd (second-order)",
}

st.set_page_config("ToM-SAR Replay", layout="wide",
                   initial_sidebar_state="expanded")

# ───────────────────── sidebar: refresh + hierarchical picker ──────────────
st.sidebar.header("Replay controls")

# ❶  Refresh button — rescans disk & clears caches
if st.sidebar.button("↻ Refresh experiments"):
    utils.EXPERIMENTS = utils._discover_experiments()     # update mapping
    load_game.cache_clear()
    pdf_to_svg.cache_clear()

exp_labels = list(utils.EXPERIMENTS.keys())
if not exp_labels:
    st.sidebar.error("No experiments found under the data directory!")
    st.stop()

# helper: split "model/exp_name/seedX" → (model, exp_name, seed)
def split_label(label: str):
    parts = label.split("/")
    return parts[0], parts[1], parts[2] if len(parts) > 2 else ("", "", "")

# ── dropdown 1: model ──────────────────────────────────────────────────────
models = sorted({split_label(lbl)[0] for lbl in exp_labels})
if "model_sel" not in st.session_state or st.session_state.model_sel not in models:
    st.session_state.model_sel = models[0]
model_sel = st.sidebar.selectbox("Model", models, key="model_sel")

# ── dropdown 2: experiment name (depend on chosen model) ───────────────────
exps_for_model = sorted({split_label(lbl)[1]
                         for lbl in exp_labels if lbl.startswith(model_sel + "/")})
if "exp_sel" not in st.session_state or st.session_state.exp_sel not in exps_for_model:
    st.session_state.exp_sel = exps_for_model[0]
exp_sel = st.sidebar.selectbox("Experiment", exps_for_model, key="exp_sel")

# ── dropdown 3: seed (depend on model+experiment) ──────────────────────────
seed_prefix = f"{model_sel}/{exp_sel}/"
seeds_for_exp = sorted({split_label(lbl)[2]
                        for lbl in exp_labels if lbl.startswith(seed_prefix)})
if "seed_sel" not in st.session_state or st.session_state.seed_sel not in seeds_for_exp:
    st.session_state.seed_sel = seeds_for_exp[0]
seed_sel = st.sidebar.selectbox("Seed", seeds_for_exp, key="seed_sel")

# ── compose the final label and store in session_state.exp ─────────────────
st.session_state.exp = f"{model_sel}/{exp_sel}/{seed_sel}"
exp_dir = utils.EXPERIMENTS[st.session_state.exp]

# ── read args.json (silently ignore if missing) ───────────────────────────
args_path = exp_dir / "args.json"
args_dict = {}
if args_path.exists():
    with open(args_path, "r", encoding="utf-8") as f:
        args_dict = json.load(f)
        
# ── read results.json (silently ignore if missing) ───────────────────────────
results_path = exp_dir / "results.json"
results_dict = {}
if results_path.exists():
    with open(results_path, "r", encoding="utf-8") as f:
        results_dict = json.load(f)

# ───────────────────── sidebar: round slider & autoplay ─────────────────────
try:    
    df = load_game(st.session_state.exp)
except:
    try:
        df = load_game(st.session_state.exp, source="csv")
    except FileNotFoundError as e:
        st.sidebar.error(f"Error loading experiment data: {e}")
        st.stop()

round_ids = sorted(df["round"].unique())

if "round" not in st.session_state or st.session_state.round not in round_ids:
    st.session_state.round = round_ids[0]

st.sidebar.slider("Round", min_value=round_ids[0], max_value=round_ids[-1],
                  key="round", format="%d")
st.sidebar.checkbox("Auto-play", key="auto")
speed = st.sidebar.slider("Speed (sec / round)", 0.3, 3.0, 1.0, 0.1)

with st.sidebar.expander("Experiment args", expanded=False):
    if args_dict:
        for k, v in sorted(args_dict.items()):
            st.markdown(f"**{k}**: {v}")
    else:
        st.markdown("_args.json not found_")

percentage_features = ['action_success_rate', 'valid_action_rate',]

with st.sidebar.expander("Final Results", expanded=False):
    if results_dict:
        for k, v in results_dict.items():
            if k in percentage_features:
                v = f"{v:.2%}" if isinstance(v, float) else v
            st.markdown(f"**{k}**: {v}")
    else:
        st.markdown("_results.json not found_")

# ─────────────────────────── cost estimate panel ─────────────────────────
model_name = args_dict.get("model_name", args_dict.get("model", ""))
with st.sidebar.expander("Price estimate (USD)", expanded=False):
    if not results_dict:
        st.markdown("_results.json not found_")
    elif not model_name:
        st.markdown("_No model name found in args.json_")
    elif "prompt_tokens" not in results_dict or "completion_tokens" not in results_dict:
        st.markdown("_Token data not found in results.json_")
    else:
        price_row = match_price_row(model_name, load_pricing())
        if price_row:
            inp, cache, out, total = estimate_cost(results_dict, price_row)
            st.metric("Estimated total", f"${total}")
            st.markdown(
                f"- **Non-cached Input:**  {results_dict.get('prompt_tokens',0)-results_dict.get('prompt_tokens_details.cached_tokens',0):,}  →  ${inp}\n"
                f"- **Cached Input:**  {results_dict.get('prompt_tokens_details.cached_tokens',0):,}  →  ${cache}\n"
                f"- **Output:** {results_dict.get('completion_tokens',0):,}  →  ${out}"
            )
            st.caption(f"Model: **{price_row['Model']}**  "
                       f"(rates per 1M — in: {price_row['Input']}, "
                       f"cached: {price_row['Cached input']}, out: {price_row['Output']})")
        else:
            st.markdown("_Model not found in pricing table_")

# ───────────────────────── main layout ─────────────────────
left, right = st.columns([2, 3])
sel_round   = st.session_state.round

# ── Chat & metrics ─────────────────────────────────────────
with left:
    st.markdown(f"### Round {sel_round}")
    this = df[df["round"] == sel_round]

    # Get score from the last match for the corresponding round
    last_row = this.iloc[-1]
    score = extract_score(last_row["obs_text"])
    if score is not None:
        st.metric("Team score", score)

    for _, row in this.iterrows():
        agent, color = row["agent_id"], AGENT_COLORS.get(row["agent_id"], "#777")
        with st.chat_message(agent, avatar="🧑‍🚒"):
            st.markdown(f"<span style='color:{color};font-weight:bold'>{agent}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Action:** {row['action']}")
            msg = row["comm"]
            st.markdown(f"**Message:** {msg}" if isinstance(msg, str) and msg.strip()
                        else "_(no message)_")

        if pd.notna(row.get("new_belief", None)):
            with st.expander(f"{agent} belief state"):
                st.write(row["new_belief"])

        def truth_chip(val: bool | None) -> str:
            if val is True:
                return '<span style="color:#3cb44b;font-weight:bold">✅ Yes</span>'
            if val is False:
                return '<span style="color:#e6194b;font-weight:bold">❌ No</span>'
            return '<span style="color:#777">❓ Unknown</span>'

        for lvl in ("ToM1st", "ToM2nd", "ToM3rd"):
            ans = row.get(lvl)
            if pd.notna(ans):                                # only show if answer exists
                q   = row.get(f"{lvl}_q", "(question not recorded)")
                gt  = row.get("ground_truth", None)          # same for all three levels
                lab = TOM_LABELS[lvl]                       # pretty label defined earlier

                with st.expander(f"{agent} {lab}"):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"**A:** {ans}")
                    st.markdown(f"**Ground Truth:** {truth_chip(gt)}", unsafe_allow_html=True)


# ── Vector map ─────────────────────────────────────────────
with right:
    svg_xml = expand_svg(Path(pdf_to_svg(st.session_state.exp, sel_round)).read_text())
    st.image(svg_xml, output_format="svg", use_container_width=True)

# ───────────────────────── autoplay scheduler ───────────────────────────
if st.session_state.get("auto"):
    time.sleep(speed)
    next_idx = (round_ids.index(sel_round) + 1) % len(round_ids)
    st.session_state.round_pending = round_ids[next_idx]   # schedule for next run
    st.rerun()
