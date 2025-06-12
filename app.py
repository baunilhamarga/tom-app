"""
Streamlit replay viewer for the EMNLP-23 ToM search-and-rescue game.
Run with:  streamlit run app.py
"""

import time, re, streamlit as st, pandas as pd
from pathlib import Path
import utils                         # we’ll mutate utils.EXPERIMENTS
from utils import load_game, pdf_to_svg, expand_svg

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

# ───────────────────── sidebar: refresh + experiment picker ─────────────────
st.sidebar.header("Replay controls")

# ❶  Refresh button — rescans disk & clears caches
if st.sidebar.button("↻ Refresh experiments"):
    utils.EXPERIMENTS = utils._discover_experiments()   # update mapping
    load_game.cache_clear()
    pdf_to_svg.cache_clear()

# ❷  Dropdown populated from (possibly refreshed) mapping
exp_labels = list(utils.EXPERIMENTS.keys())
if not exp_labels:
    st.sidebar.error("No experiments found under the data directory!")
    st.stop()

if "exp" not in st.session_state or st.session_state.exp not in exp_labels:
    st.session_state.exp = exp_labels[0]

st.sidebar.selectbox("Experiment", exp_labels, key="exp",
                     format_func=lambda x: x)           # show path as-is

# ───────────────────── sidebar: round slider & autoplay ─────────────────────
df = load_game(st.session_state.exp)
round_ids = sorted(df["round"].unique())

if "round" not in st.session_state or st.session_state.round not in round_ids:
    st.session_state.round = round_ids[0]

st.sidebar.slider("Round", min_value=round_ids[0], max_value=round_ids[-1],
                  key="round", format="%d")
st.sidebar.checkbox("Auto-play", key="auto")
speed = st.sidebar.slider("Speed (sec / round)", 0.3, 3.0, 1.0, 0.1)

# ───────────────────────── main layout ─────────────────────
left, right = st.columns([2, 3])
sel_round   = st.session_state.round

# ── Chat & metrics ─────────────────────────────────────────
with left:
    st.markdown(f"### Round {sel_round}")
    this = df[df["round"] == sel_round]

    score = extract_score(this.iloc[0]["obs_text"])
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
