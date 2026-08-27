"""
dataset_pipeline.py
---------------------
Analyzes the RAW dataset (data/unified_dataset_clean.csv) directly —
no more relying on pre-computed analysis_summary.json / ab_test_results.json.

Every time this runs it:
  1. Loads the current CSV as-is.
  2. Computes descriptive stats, emotion distribution, engagement/attention
     by stimulus variant, K-means segmentation, and pairwise A/B/C t-tests
     — straight from the rows.
  3. Compares against the LAST snapshot to figure out exactly which rows
     are new since last time, and summarizes what changed.
  4. Appends the result to reports/dashboard_history.jsonl.

Drop into: Neuromarketing_project/src/dataset_pipeline.py

Expected CSV columns (yours):
  user_id, emotion, user_id.1, gaze_x, gaze_y, attention_time,
  emotion_score, engagement_score, Product Features, Eye direction

- 'emotion' / 'emotion_score' / 'attention_time' / 'engagement_score' are
  present for every row (general trials).
- 'Eye direction' (A (Left) / B (Right) / C (Neutral)) and
  'Product Features' are only populated for the subset of rows that belong
  to the stimulus A/B/C comparison test.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATASET_PATH = os.path.join("data", "unified_dataset_clean.csv")
HISTORY_PATH = os.path.join("reports", "dashboard_history.jsonl")

REQUIRED_COLS = ["user_id", "emotion", "attention_time", "emotion_score", "engagement_score"]


# ---------------------------------------------------------------------
# 1. LOAD + VALIDATE
# ---------------------------------------------------------------------
def load_dataset(csv_path: str = DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def load_snapshot_history(history_path: str = HISTORY_PATH) -> list:
    if not os.path.exists(history_path):
        return []
    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------
# 2. CORE ANALYSIS — computed directly from the dataframe
# ---------------------------------------------------------------------
def _round_dict(d, ndigits=3):
    return {k: (round(v, ndigits) if isinstance(v, (int, float, np.floating)) else v) for k, v in d.items()}


def analyze_dataset(df: pd.DataFrame) -> dict:
    n_trials = len(df)
    n_participants = df["user_id"].nunique()

    emotion_distribution = _round_dict(df["emotion"].value_counts(normalize=True).to_dict(), 4)
    avg_engagement_score = round(df["engagement_score"].mean(), 3)
    avg_attention_time = round(df["attention_time"].mean(), 3)

    # --- stimulus subset (only rows with an assigned variant) ---
    stim_df = df[df.get("Eye direction").notna()] if "Eye direction" in df.columns else df.iloc[0:0]
    engagement_by_stimulus = _round_dict(
        stim_df.groupby("Eye direction")["engagement_score"].mean().to_dict(), 3
    ) if not stim_df.empty else {}
    attention_by_stimulus = _round_dict(
        stim_df.groupby("Eye direction")["attention_time"].mean().to_dict(), 3
    ) if not stim_df.empty else {}

    # --- segmentation (K-means on all trials with complete features) ---
    seg_cols = ["emotion_score", "attention_time", "engagement_score"]
    seg_data = df[seg_cols].dropna()
    segment_profiles, segment_sizes = {}, {}
    if len(seg_data) >= 3:
        X = StandardScaler().fit_transform(seg_data)
        k = min(3, len(seg_data))
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        seg_data = seg_data.copy()
        seg_data["cluster"] = km.labels_
        for cluster_id, group in seg_data.groupby("cluster"):
            cid = str(cluster_id)
            segment_sizes[cid] = int(len(group))
            segment_profiles[cid] = {
                "avg_emotion_score": round(group["emotion_score"].mean(), 3),
                "avg_attention_time": round(group["attention_time"].mean(), 3),
                "avg_engagement_score": round(group["engagement_score"].mean(), 3),
            }

    # --- A/B/C pairwise significance testing ---
    ab_test_results = {}
    variants = sorted(stim_df["Eye direction"].dropna().unique().tolist()) if not stim_df.empty else []
    for metric in ["engagement_score", "emotion_score", "attention_time"]:
        comparisons = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                a, b = variants[i], variants[j]
                group_a = stim_df[stim_df["Eye direction"] == a][metric].dropna()
                group_b = stim_df[stim_df["Eye direction"] == b][metric].dropna()
                if len(group_a) >= 3 and len(group_b) >= 3:
                    if group_a.std() == 0 and group_b.std() == 0:
                        # No variation in either group -> a t-test is undefined, not "no difference"
                        comparisons.append({
                            "variant_a": a, "variant_b": b,
                            "mean_a": round(group_a.mean(), 3), "mean_b": round(group_b.mean(), 3),
                            "note": "no variation in this metric for either variant (identical values) — significance cannot be computed",
                            "n_a": int(len(group_a)), "n_b": int(len(group_b)),
                        })
                        continue
                    tstat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
                    significant = bool(p_value < 0.05)
                    if significant:
                        better = a if group_a.mean() > group_b.mean() else b
                    else:
                        better = "no significant difference"
                    comparisons.append({
                        "variant_a": a, "variant_b": b,
                        "mean_a": round(group_a.mean(), 3), "mean_b": round(group_b.mean(), 3),
                        "p_value": round(float(p_value), 4),
                        "significant": significant,
                        "likely_better_variant": better,
                        "n_a": int(len(group_a)), "n_b": int(len(group_b)),
                    })
                else:
                    comparisons.append({
                        "variant_a": a, "variant_b": b,
                        "note": "insufficient samples (need >= 3 per variant)",
                        "n_a": int(len(group_a)), "n_b": int(len(group_b)),
                    })
        ab_test_results[metric] = comparisons

    return {
        "n_trials": n_trials,
        "n_participants": int(n_participants),
        "emotion_distribution": emotion_distribution,
        "avg_engagement_score": avg_engagement_score,
        "avg_attention_time": avg_attention_time,
        "engagement_by_stimulus": engagement_by_stimulus,
        "attention_by_stimulus": attention_by_stimulus,
        "segment_sizes": segment_sizes,
        "segment_profiles": segment_profiles,
        "ab_test_results": ab_test_results,
    }


# ---------------------------------------------------------------------
# 3. DIFF: exactly which rows are new since the last analysis run
# ---------------------------------------------------------------------
def _describe_new_rows(df: pd.DataFrame, prev_n_trials: int) -> str:
    """Assumes new data is appended to the end of the CSV (typical for a
    growing collection log). Slices out only the new rows and summarizes them."""
    new_df = df.iloc[prev_n_trials:]
    if new_df.empty:
        return "No new rows since the last analysis."
    n_new = len(new_df)
    new_participants = new_df["user_id"].nunique()
    dom_emotion = new_df["emotion"].mode().iloc[0] if not new_df["emotion"].mode().empty else "N/A"
    avg_eng_new = new_df["engagement_score"].mean()
    return (
        f"{n_new} new row(s) added, from {new_participants} participant(s). "
        f"Among just the new data, the dominant emotion was **{dom_emotion}** "
        f"and average engagement score was {avg_eng_new:.2f}."
    )


def _generate_explanation(curr: dict, prev: dict | None, new_rows_text: str | None) -> str:
    parts = []
    if prev is None:
        parts.append(
            f"First recorded analysis: {curr['n_trials']} total trials across "
            f"{curr['n_participants']} participants."
        )
    else:
        if new_rows_text:
            parts.append(new_rows_text)
        prev_eng, curr_eng = prev.get("avg_engagement_score"), curr.get("avg_engagement_score")
        if prev_eng is not None and curr_eng is not None:
            diff = curr_eng - prev_eng
            pct = (diff / prev_eng * 100) if prev_eng else 0
            direction = "up" if diff > 0 else ("down" if diff < 0 else "unchanged")
            parts.append(
                f"Overall average engagement score moved {direction} from {prev_eng:.2f} "
                f"to {curr_eng:.2f} ({pct:+.1f}%) across the whole dataset."
            )

    eng_by_stim = curr.get("engagement_by_stimulus", {})
    if eng_by_stim:
        best = max(eng_by_stim, key=eng_by_stim.get)
        worst = min(eng_by_stim, key=eng_by_stim.get)
        parts.append(f"**{best}** currently leads in engagement ({eng_by_stim[best]:.2f}) vs **{worst}** ({eng_by_stim[worst]:.2f}).")

    emo_dist = curr.get("emotion_distribution", {})
    if emo_dist:
        top_emotion = max(emo_dist, key=emo_dist.get)
        parts.append(f"Most frequent emotion overall: **{top_emotion}** ({emo_dist[top_emotion]*100:.1f}%).")

    sig = []
    for metric_name, comparisons in curr.get("ab_test_results", {}).items():
        for comp in comparisons:
            if comp.get("significant"):
                sig.append(f"{metric_name.replace('_',' ')}: {comp['likely_better_variant']} (p={comp['p_value']})")
    parts.append("Significant A/B findings: " + "; ".join(sig) + "." if sig else
                 "No statistically significant A/B differences yet.")

    return " ".join(parts)


# ---------------------------------------------------------------------
# 4. MAIN ENTRY POINT — call this after your dataset is updated
# ---------------------------------------------------------------------
def capture_snapshot_from_dataset(csv_path: str = DATASET_PATH, history_path: str = HISTORY_PATH) -> dict:
    df = load_dataset(csv_path)
    history = load_snapshot_history(history_path)
    prev = history[-1] if history else None

    analysis = analyze_dataset(df)

    new_rows_text = None
    if prev is not None:
        prev_n = prev.get("n_trials", 0)
        if analysis["n_trials"] > prev_n:
            new_rows_text = _describe_new_rows(df, prev_n)
        elif analysis["n_trials"] == prev_n:
            new_rows_text = "No new rows since the last analysis — dataset size unchanged."

    snapshot = dict(analysis)
    snapshot["timestamp"] = datetime.now().isoformat(timespec="seconds")
    snapshot["explanation"] = _generate_explanation(analysis, prev, new_rows_text)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")

    return snapshot


def has_new_data(csv_path: str = DATASET_PATH, history_path: str = HISTORY_PATH) -> bool:
    """Cheap check: does the CSV currently have a different row count than
    the last recorded snapshot? Used by the dashboard to auto-refresh."""
    if not os.path.exists(csv_path):
        return False
    current_n = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1  # minus header
    history = load_snapshot_history(history_path)
    if not history:
        return True
    return current_n != history[-1].get("n_trials")


if __name__ == "__main__":
    snap = capture_snapshot_from_dataset()
    print("Snapshot captured directly from dataset:\n")
    print(snap["explanation"])