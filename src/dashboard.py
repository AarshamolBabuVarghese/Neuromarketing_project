"""
dashboard.py
---------------------------------------------------------------
BI Dashboard for the Neuromarketing Project.

This version adds the genuinely useful BI features from the old
dataset_pipeline.py-based dashboard (auto-refresh on new data,
snapshot history, trend charts, previous-vs-current comparison,
optional participant clustering) WITHOUT bringing back the one
thing that was actually broken there: a global "A/B Test Results"
table that compared Eye direction A vs B vs C across the ENTIRE
dataset, mixing every different test together. That confound is
exactly what rec.py (ab_recommendation_engine.py) exists to avoid,
so all real A/B winners here still come only from rec.py's already
correct, per-test-grouped output -- never recomputed globally.

Run (inside venv_dashboard):
  streamlit run src/dashboard.py
  (clustering section needs: pip install scikit-learn)

Reads:
  data/stimulus_manifest.csv                                       (category mapping)
  data/unified_dataset_clean.csv                                    (descriptive stats)
  results/ab_recommendations/ab_recommendations.csv                 (per-test results, from rec.py)
  results/ab_recommendations/product_reports/product_level_recommendations.csv

Writes (its own, separate from the old project's history file):
  results/ab_recommendations/dashboard_history.jsonl
---------------------------------------------------------------
"""
import os
import json
import time
import pandas as pd
import altair as alt
import streamlit as st

MANIFEST_PATH = os.path.join("data", "stimulus_manifest.csv")
DATASET_PATH = os.path.join("data", "unified_dataset_clean.csv")
RAW_CSV = os.path.join("results", "ab_recommendations", "ab_recommendations.csv")
PRODUCT_CSV = os.path.join("results", "ab_recommendations", "product_reports",
                            "product_level_recommendations.csv")
HISTORY_PATH = os.path.join("results", "ab_recommendations", "dashboard_history.jsonl")

st.set_page_config(page_title="Neuromarketing Analysis Dashboard", layout="wide", page_icon="🧠")
st.title("🧠 Neuromarketing Analysis Dashboard")
st.caption("Descriptive stats from the raw dataset + attributable A/B recommendations from rec.py")


def render_donut(counts: pd.Series, height: int = 300, color_scheme: str = None):
    """
    Renders a donut chart from a pandas Series of label -> count
    (e.g. the output of .value_counts()). Uses altair, which is
    already installed as a streamlit dependency -- no extra pip
    install needed.
    """
    df = counts.reset_index()
    df.columns = ["label", "value"]
    color = alt.Color("label:N", legend=alt.Legend(title=None))
    if color_scheme:
        color = alt.Color("label:N", legend=alt.Legend(title=None), scale=alt.Scale(scheme=color_scheme))
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=60)
        .encode(theta="value:Q", color=color, tooltip=["label", "value"])
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
@st.cache_data
def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None
    m = pd.read_csv(MANIFEST_PATH)
    lookup = {}
    for _, row in m.iterrows():
        cat = row.get("category")
        lookup[row["test_name"]] = cat
        if pd.notna(row.get("product_name")) and pd.notna(row.get("attribute_changed")):
            lookup[f"{row['product_name']}_{row['attribute_changed']}"] = cat
    return lookup


@st.cache_data
def load_raw_dataset(_manifest_lookup):
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    if _manifest_lookup and "Product Features" in df.columns:
        df["category"] = df["Product Features"].map(_manifest_lookup)
    else:
        df["category"] = None
    return df


@st.cache_data
def load_recommendations():
    if not os.path.exists(RAW_CSV):
        return None, None
    raw = pd.read_csv(RAW_CSV)
    product = pd.read_csv(PRODUCT_CSV) if os.path.exists(PRODUCT_CSV) else pd.DataFrame()
    return raw, product


def file_last_modified(path):
    if not os.path.exists(path):
        return None
    return time.strftime("%d %b %Y, %H:%M", time.localtime(os.path.getmtime(path)))


# ------------------------------------------------------------------
# Snapshot history -- OUR OWN, separate from the old project's
# reports/dashboard_history.jsonl. Only ever records safe, non-
# confounded descriptive stats plus rec.py's own confidence counts.
# ------------------------------------------------------------------
def load_snapshot_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def capture_snapshot(dataset_df, rec_df):
    snapshot = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_trials": int(len(dataset_df)),
        "n_participants": int(dataset_df["user_id"].nunique()) if "user_id" in dataset_df.columns else None,
        "avg_engagement_score": float(dataset_df["engagement_score"].mean())
            if "engagement_score" in dataset_df.columns and len(dataset_df) else None,
        "avg_attention_time": float(dataset_df["attention_time"].mean())
            if "attention_time" in dataset_df.columns and len(dataset_df) else None,
        "emotion_distribution": dataset_df["emotion"].value_counts(normalize=True).round(3).to_dict()
            if "emotion" in dataset_df.columns and len(dataset_df) else {},
        "n_high_confidence": int((rec_df["confidence"] == "High").sum()) if rec_df is not None else None,
        "n_medium_confidence": int((rec_df["confidence"] == "Medium").sum()) if rec_df is not None else None,
        "n_inconclusive": int((rec_df["winner_variant"] == "No clear winner").sum()) if rec_df is not None else None,
    }
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
    return snapshot


manifest_lookup = load_manifest()
dataset_df = load_raw_dataset(manifest_lookup)
rec_df, product_df = load_recommendations()

if dataset_df is None:
    st.error(f"Couldn't find `{DATASET_PATH}`.")
    st.stop()

# ------------------------------------------------------------------
# Auto-detect new data: if the current trial count differs from the
# last recorded snapshot, capture a new one automatically.
# ------------------------------------------------------------------
history = load_snapshot_history()
current_n_trials = len(dataset_df)
just_updated = False

if not history or history[-1]["n_trials"] != current_n_trials:
    capture_snapshot(dataset_df, rec_df)
    history = load_snapshot_history()
    just_updated = True

if just_updated and len(history) > 1:
    st.success("📊 New data was detected and a new snapshot has been recorded automatically.")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
if manifest_lookup:
    all_categories = sorted({c for c in manifest_lookup.values() if pd.notna(c)})
elif rec_df is not None:
    all_categories = sorted(rec_df["category"].dropna().unique().tolist())
else:
    all_categories = []

with st.sidebar:
    st.header("Browse by category")
    selected_categories = st.multiselect("Category", options=all_categories, default=all_categories)
    confidence_filter = st.multiselect(
        "Confidence level", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_b:
        if st.button("📸 New snapshot", use_container_width=True):
            capture_snapshot(dataset_df, rec_df)
            st.rerun()
    rec_updated = file_last_modified(RAW_CSV)
    st.caption(
        f"Recommendations last generated: {rec_updated or 'never — run rec.py'}\n\n"
        f"If you've collected more participants since then, re-run "
        f"`python src/rec.py` and `python src/product_report_rebuilder.py` "
        f"in your data-collection venv, then click Refresh above."
    )

dataset_filtered = dataset_df[dataset_df["category"].isin(selected_categories)] \
    if selected_categories else dataset_df.iloc[0:0]

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trials (filtered)", f"{len(dataset_filtered):,}")
col2.metric("Participants (filtered)",
            f"{dataset_filtered['user_id'].nunique():,}" if len(dataset_filtered) else "0")
col3.metric("Avg Engagement Score",
            f"{dataset_filtered['engagement_score'].mean():.2f}" if len(dataset_filtered) else "N/A")
col4.metric("High-Confidence Recommendations",
            int((rec_df["confidence"] == "High").sum()) if rec_df is not None else 0)

if rec_df is not None and len(rec_df):
    with st.expander("📦 Tests by Category (overview)", expanded=False):
        render_donut(rec_df["category"].value_counts(), height=280, color_scheme="tableau10")

# ------------------------------------------------------------------
# Previous vs Current comparison (uses OUR OWN history, unfiltered
# by category so the trend reflects the whole project over time)
# ------------------------------------------------------------------
if len(history) > 1:
    prev, curr = history[-2], history[-1]
    st.subheader("🔍 Previous vs. Current Snapshot — What Changed")
    rows = []
    for label, key, fmt in [
        ("Total Trials", "n_trials", "{:,.0f}"),
        ("Total Participants", "n_participants", "{:,.0f}"),
        ("Avg Engagement Score", "avg_engagement_score", "{:.2f}"),
        ("Avg Attention Time (s)", "avg_attention_time", "{:.2f}"),
        ("High-Confidence Recommendations", "n_high_confidence", "{:,.0f}"),
        ("Inconclusive Tests", "n_inconclusive", "{:,.0f}"),
    ]:
        pv, cv = prev.get(key), curr.get(key)
        if pv is not None and cv is not None:
            change = cv - pv
            arrow = "🔼" if change > 0 else ("🔽" if change < 0 else "➖")
            rows.append({"Metric": label, "Previous": fmt.format(pv), "Current": fmt.format(cv),
                         "Change": f"{arrow} {fmt.format(abs(change))}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.divider()

# ------------------------------------------------------------------
# Trend charts over analysis history
# ------------------------------------------------------------------
if len(history) > 1:
    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], format="ISO8601")
    hist_df = hist_df.set_index("timestamp")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**📈 Dataset Growth (Trials per Snapshot)**")
        st.line_chart(hist_df["n_trials"])
    with t2:
        st.markdown("**📊 Avg Engagement Over Time**")
        st.line_chart(hist_df["avg_engagement_score"])
    with t3:
        st.markdown("**✅ High-Confidence Recommendations Over Time**")
        st.line_chart(hist_df["n_high_confidence"])
    st.divider()
else:
    st.info("📊 Trend charts need at least 2 snapshots. Come back after collecting more data "
             "(a new snapshot is captured automatically whenever the trial count changes).")

# ------------------------------------------------------------------
# Descriptive-only charts -- filtered by category.
# ------------------------------------------------------------------
st.subheader("Current Snapshot Breakdown")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**😊 Emotion Distribution**")
    if "emotion" in dataset_filtered.columns and len(dataset_filtered):
        render_donut(dataset_filtered["emotion"].value_counts())
    else:
        st.caption("No data for the selected category.")

with c2:
    st.markdown("**💡 Avg Engagement by Stimulus**")
    if {"Product Features", "engagement_score"}.issubset(dataset_filtered.columns) and len(dataset_filtered):
        eng = dataset_filtered.groupby("Product Features")["engagement_score"].mean().sort_values(ascending=False)
        st.bar_chart(eng)
    else:
        st.caption("No data for the selected category.")

with c3:
    st.markdown("**👁️ Avg Attention Time by Stimulus**")
    if {"Product Features", "attention_time"}.issubset(dataset_filtered.columns) and len(dataset_filtered):
        att = dataset_filtered.groupby("Product Features")["attention_time"].mean().sort_values(ascending=False)
        st.bar_chart(att)
    else:
        st.caption("No data for the selected category.")

st.divider()

# ------------------------------------------------------------------
# Optional: Participant Segments (Clustering) -- exploratory only.
# Clearly labeled as NOT a product recommendation, since clustering
# raw behavioral features has no connection to "which attribute won".
# Needs scikit-learn (pip install scikit-learn in venv_dashboard).
# ------------------------------------------------------------------
with st.expander("🧩 Participant Segments (exploratory clustering, not a recommendation)"):
    st.caption(
        "Groups participants by raw behavior (engagement, emotion, attention) using "
        "k-means. This is exploratory audience profiling -- it does NOT tell you which "
        "A/B variant wins; only the per-product recommendations above do that."
    )
    try:
        from sklearn.cluster import KMeans
        feature_cols = [c for c in ["engagement_score", "emotion_score", "attention_time"]
                         if c in dataset_filtered.columns]
        cluster_data = dataset_filtered[feature_cols].dropna()
        if len(cluster_data) >= 6:
            n_clusters = min(3, len(cluster_data) // 2)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = km.fit_predict(cluster_data)
            cluster_data = cluster_data.copy()
            cluster_data["segment"] = labels
            profile = cluster_data.groupby("segment")[feature_cols].mean().round(2)
            sizes = cluster_data["segment"].value_counts().rename("size")
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                st.dataframe(profile.join(sizes), use_container_width=True)
            with sc2:
                st.markdown("**Segment Sizes**")
                sizes_named = sizes.copy()
                sizes_named.index = [f"Segment {i}" for i in sizes_named.index]
                render_donut(sizes_named, height=250)
        else:
            st.caption("Not enough filtered data yet to form meaningful segments (need 6+ trials).")
    except ImportError:
        st.caption("Install scikit-learn to enable this section: `pip install scikit-learn`")

st.divider()

# ------------------------------------------------------------------
# A/B Recommendations -- sourced from rec.py's per-test grouped
# output, filtered by category. This is the ONLY place actual A/B
# winners are shown -- never recomputed globally on this page.
# ------------------------------------------------------------------
st.subheader("🆚 A/B Recommendations (per product, per attribute)")

if rec_df is None:
    st.warning(
        f"Couldn't find `{RAW_CSV}`. Run `python src/rec.py` and "
        f"`python src/product_report_rebuilder.py` in your data-collection "
        f"venv first, then click Refresh in the sidebar."
    )
    st.stop()

filtered = rec_df[
    rec_df["category"].isin(selected_categories) & rec_df["confidence"].isin(confidence_filter)
]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products shown", filtered["product_name"].nunique())
k2.metric("Tests shown", len(filtered))
k3.metric("High confidence", int((filtered["confidence"] == "High").sum()))
k4.metric("Inconclusive", int((filtered["winner_variant"] == "No clear winner").sum()))

conf_counts = filtered["confidence"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
cc1, cc2 = st.columns(2)
with cc1:
    st.bar_chart(conf_counts)
with cc2:
    render_donut(conf_counts[conf_counts > 0], height=250)

BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "⚪ Low"}

if filtered.empty:
    st.info("No tests match the current filters. If you just collected data for a "
             "new category, make sure you've re-run rec.py + product_report_rebuilder.py, "
             "then click Refresh in the sidebar.")
else:
    high_counts = filtered[filtered["confidence"] == "High"].groupby("product_name").size().to_dict()
    product_order = sorted(filtered["product_name"].unique(), key=lambda p: -high_counts.get(p, 0))

    for product_name in product_order:
        group = filtered[filtered["product_name"] == product_name]
        category = group["category"].iloc[0]
        n_high = (group["confidence"] == "High").sum()
        n_med = (group["confidence"] == "Medium").sum()

        header = f"**{product_name}**  ·  {category}  ·  {len(group)} attribute(s) tested"
        if n_high:
            header += f"  ·  🟢 {n_high} high-confidence"
        if n_med:
            header += f"  ·  🟡 {n_med} to consider"

        with st.expander(header, expanded=bool(n_high)):
            for _, row in group.sort_values(
                "confidence", key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2})
            ).iterrows():
                badge = BADGE.get(row["confidence"], row["confidence"])
                attribute = str(row["attribute_changed"]).replace("_", " ").title()
                if row["winner_variant"] == "No clear winner":
                    st.markdown(f"- {badge} — **{attribute}**: no significant preference found yet.")
                else:
                    st.markdown(f"- {badge} — **{attribute}**: adopt **{row.get('winner_value', '')}**")
                    st.caption(row.get("recommendation", ""))

st.divider()

# ------------------------------------------------------------------
# Analysis History Feed
# ------------------------------------------------------------------
st.subheader("🗂️ Snapshot History")
for snap in reversed(history):
    ts = pd.to_datetime(snap["timestamp"])
    label = f"{ts.strftime('%d %b %Y, %H:%M')}  ·  {snap['n_trials']} trials  ·  " \
            f"{snap.get('n_high_confidence', 0)} high-confidence"
    with st.expander(label, expanded=(snap is history[-1])):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trials", snap["n_trials"])
        m2.metric("Participants", snap.get("n_participants", "N/A"))
        m3.metric("Avg Engagement", f"{snap['avg_engagement_score']:.2f}" if snap.get("avg_engagement_score") else "N/A")
        m4.metric("High Confidence", snap.get("n_high_confidence", "N/A"))

with st.expander("Raw recommendation data (filtered)"):
    st.dataframe(filtered, use_container_width=True)