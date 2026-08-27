"""dashboard.py
-------------
BI Dashboard for the Neuromarketing Project — built directly on top of
your REAL, existing analysis_summary.json + ab_test_results.json.

Run with:  streamlit run src/dashboard.py

How it works:
  - Your pipeline (main.py) overwrites analysis_summary.json / ab_test_results.json
    every time it runs, with cumulative stats (n_trials grows, averages update).
  - This dashboard doesn't just show the latest file — it keeps a SNAPSHOT
    HISTORY (reports/dashboard_history.jsonl) so you can see how the analysis
    has evolved: how many new data entries came in, how averages shifted,
    which variant is winning now vs before.
  - Each snapshot has an auto-generated plain-English explanation.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from dataset_pipeline import (  # noqa: E402
    DATASET_PATH, HISTORY_PATH, capture_snapshot_from_dataset,
    has_new_data, load_snapshot_history,
)

st.set_page_config(page_title="Neuromarketing BI Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 Neuromarketing Analysis Dashboard")
st.caption("Eye-tracking + facial emotion recognition — analyzed directly from data/unified_dataset_clean.csv")

# ---------------------------------------------------------------------
# Auto-detect new data & re-analyze automatically (no manual click needed)
# ---------------------------------------------------------------------
just_updated = False
if os.path.exists(DATASET_PATH) and has_new_data(DATASET_PATH, HISTORY_PATH):
    with st.spinner("New data detected in dataset — running fresh analysis..."):
        capture_snapshot_from_dataset(DATASET_PATH, HISTORY_PATH)
    just_updated = True

with st.sidebar:
    st.header("Controls")
    st.write(
        "The dashboard automatically re-analyzes `data/unified_dataset_clean.csv` "
        "whenever the row count changes — just reload this page after adding data. "
        "Use the button below to force a re-analysis even if the row count is the same "
        "(e.g. if existing values were edited)."
    )
    if st.button("🔄 Force re-analyze dataset now", use_container_width=True):
        if not os.path.exists(DATASET_PATH):
            st.error(f"{DATASET_PATH} not found.")
        else:
            with st.spinner("Analyzing dataset..."):
                capture_snapshot_from_dataset(DATASET_PATH, HISTORY_PATH)
            st.success("Re-analyzed!")
            st.rerun()

if just_updated:
    st.success("📊 New data was detected and the analysis below has been refreshed automatically.")

# ---------------------------------------------------------------------
# Load snapshot history
# ---------------------------------------------------------------------
history = load_snapshot_history(HISTORY_PATH)

if not history:
    st.warning(
        "No snapshots recorded yet. Click **'Capture latest analysis as new "
        "snapshot'** in the sidebar to load your current analysis_summary.json "
        "/ ab_test_results.json into the dashboard for the first time."
    )
    st.stop()

df = pd.DataFrame(history)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
latest = df.iloc[-1]

# ---------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Entries (Trials)", f"{int(latest['n_trials']):,}" if pd.notna(latest["n_trials"]) else "N/A")
col2.metric("Total Participants", f"{int(latest['n_participants']):,}" if pd.notna(latest["n_participants"]) else "N/A")
col3.metric("Avg Engagement Score", f"{latest['avg_engagement_score']:.2f}" if pd.notna(latest["avg_engagement_score"]) else "N/A")
col4.metric("Analysis Runs Logged", len(df))

if len(df) > 1:
    prev = df.iloc[-2]
    new_trials = int(latest["n_trials"] - prev["n_trials"]) if pd.notna(latest["n_trials"]) and pd.notna(prev["n_trials"]) else None
    if new_trials is not None:
        st.info(f"📥 **{new_trials} new data entries** were added since the previous analysis run.")

    st.subheader("🔍 Previous vs. Current — What Changed")
    compare_rows = []
    for label, key, fmt in [
        ("Total Trials", "n_trials", "{:,.0f}"),
        ("Total Participants", "n_participants", "{:,.0f}"),
        ("Avg Engagement Score", "avg_engagement_score", "{:.2f}"),
        ("Avg Attention Time (s)", "avg_attention_time", "{:.2f}"),
    ]:
        prev_val, curr_val = prev.get(key), latest.get(key)
        if pd.notna(prev_val) and pd.notna(curr_val):
            change = curr_val - prev_val
            pct = (change / prev_val * 100) if prev_val else 0
            arrow = "🔼" if change > 0 else ("🔽" if change < 0 else "➖")
            compare_rows.append({
                "Metric": label,
                "Previous": fmt.format(prev_val),
                "Current": fmt.format(curr_val),
                "Change": f"{arrow} {fmt.format(abs(change))} ({pct:+.1f}%)",
            })
    if compare_rows:
        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Trend charts: how the dataset & metrics evolved across analysis runs
# ---------------------------------------------------------------------
left, right = st.columns(2)

if len(df) < 2:
    st.info(
        "📊 Trend charts need at least 2 analysis runs to show a trend. "
        "This is your first recorded run — trends will appear once new data "
        "triggers another analysis."
    )

with left:
    st.subheader("📈 Dataset Growth (Total Trials per Analysis Run)")
    fig = px.line(df, x="timestamp", y="n_trials", markers=True)
    fig.update_layout(height=360, yaxis_title="Total Trials")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📊 Avg Engagement Score Over Analysis Runs")
    fig = px.line(df, x="timestamp", y="avg_engagement_score", markers=True)
    fig.update_layout(height=360, yaxis_title="Avg Engagement Score")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Current state: emotion distribution, engagement/attention by stimulus
# ---------------------------------------------------------------------
st.subheader("Current Snapshot Breakdown")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**😊 Emotion Distribution**")
    emo = latest.get("emotion_distribution", {})
    if emo:
        emo_df = pd.DataFrame(list(emo.items()), columns=["Emotion", "Share"])
        fig = px.pie(emo_df, names="Emotion", values="Share", hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**💡 Engagement by Stimulus**")
    eng = latest.get("engagement_by_stimulus", {})
    if eng:
        eng_df = pd.DataFrame(list(eng.items()), columns=["Stimulus", "Avg Engagement"])
        fig = px.bar(eng_df, x="Stimulus", y="Avg Engagement", color="Stimulus", text_auto=".2f")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("**👁️ Attention Time by Stimulus**")
    att = latest.get("attention_by_stimulus", {})
    if att:
        att_df = pd.DataFrame(list(att.items()), columns=["Stimulus", "Avg Attention (s)"])
        fig = px.bar(att_df, x="Stimulus", y="Avg Attention (s)", color="Stimulus", text_auto=".2f")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Segmentation (clusters)
# ---------------------------------------------------------------------
st.subheader("🧩 Participant Segments (Clustering)")
seg_profiles = latest.get("segment_profiles", {})
seg_sizes = latest.get("segment_sizes", {})
if seg_profiles:
    seg_rows = []
    for seg_id, profile in seg_profiles.items():
        row = {"Segment": seg_id, "Size": seg_sizes.get(seg_id, "N/A")}
        row.update(profile)
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        st.dataframe(seg_df, hide_index=True, use_container_width=True)
    with sc2:
        fig = px.pie(seg_df, names="Segment", values="Size", title="Segment Sizes", hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Auto-generated interpretation, same rule-based style as the rest of the dashboard
    total_participants = sum(int(v) for v in seg_sizes.values()) if seg_sizes else 0
    best_seg = max(seg_profiles, key=lambda k: seg_profiles[k].get("avg_engagement_score", 0))
    worst_seg = min(seg_profiles, key=lambda k: seg_profiles[k].get("avg_engagement_score", 0))
    best_pct = (int(seg_sizes.get(best_seg, 0)) / total_participants * 100) if total_participants else 0
    st.markdown(
        f"**Interpretation:** Segment **{best_seg}** shows the highest engagement "
        f"({seg_profiles[best_seg]['avg_engagement_score']:.2f}) and represents "
        f"{best_pct:.0f}% of all trials — this is your strongest-responding audience "
        f"profile. Segment **{worst_seg}** shows the lowest engagement "
        f"({seg_profiles[worst_seg]['avg_engagement_score']:.2f}) and may need a "
        f"different messaging or design approach."
    )

# ---------------------------------------------------------------------
# A/B Test Results (current)
# ---------------------------------------------------------------------
st.subheader("🆚 A/B Test Results (Current)")
ab_results = latest.get("ab_test_results", {})
if ab_results:
    for metric_name, comparisons in ab_results.items():
        st.markdown(f"**{metric_name.replace('_', ' ').title()}**")
        rows = []
        notes = []
        for comp in comparisons:
            has_pvalue = "p_value" in comp
            if has_pvalue:
                sig_display = "✅ Yes" if comp.get("significant") else "❌ No"
            else:
                sig_display = "⚠️ Not tested"
            rows.append({
                "Variant A": comp.get("variant_a"),
                "Variant B": comp.get("variant_b"),
                "Mean A": comp.get("mean_a", "—"),
                "Mean B": comp.get("mean_b", "—"),
                "p-value": comp.get("p_value", "—"),
                "Significant?": sig_display,
            })
            note_text = comp.get("likely_better_variant") or comp.get("note", "")
            notes.append(f"**{comp.get('variant_a')} vs {comp.get('variant_b')}:** {note_text}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("  ·  ".join(notes))

st.divider()

# ---------------------------------------------------------------------
# Analysis Feed — every snapshot, newest first, with explanation
# ---------------------------------------------------------------------
st.subheader("🗂️ Analysis History Feed")
for _, row in df.sort_values("timestamp", ascending=False).iterrows():
    label = f"Run recorded {row['timestamp'].strftime('%Y-%m-%d %H:%M')}  ·  {int(row['n_trials'])} trials"
    with st.expander(label, expanded=(row["timestamp"] == latest["timestamp"])):
        m1, m2, m3 = st.columns(3)
        m1.metric("Trials", int(row["n_trials"]) if pd.notna(row["n_trials"]) else "N/A")
        m2.metric("Participants", int(row["n_participants"]) if pd.notna(row["n_participants"]) else "N/A")
        m3.metric("Avg Engagement", f"{row['avg_engagement_score']:.2f}" if pd.notna(row["avg_engagement_score"]) else "N/A")
        st.markdown("**Explanation:**")
        st.write(row.get("explanation", "—"))