"""
analysis.py
------------
Core neuromarketing analysis:
  1. Descriptive stats: emotion distribution, avg engagement per product/stimulus
  2. KMeans segmentation: cluster participants into engagement "personas"
     (e.g. high-attention/high-emotion vs low-attention/low-emotion)
  3. Saves everything to reports/analysis_summary.json for the client report
"""
import json
import os
import yaml
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from preprocessing import run as run_preprocessing

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

REPORTS_DIR = _CFG["paths"]["reports_dir"]
N_CLUSTERS = _CFG["analysis"]["n_clusters"]


def descriptive_stats(df: pd.DataFrame) -> dict:
    emotion_dist = df["emotion"].value_counts(normalize=True).round(3).to_dict()

    gaze_df = df[df["usable_for_gaze_analysis"]]
    engagement_by_stimulus = (
        gaze_df.groupby("Eye direction")["engagement_score"]
        .mean().round(3).to_dict()
        if not gaze_df.empty else {}
    )
    attention_by_stimulus = (
        gaze_df.groupby("Eye direction")["attention_time"]
        .mean().round(3).to_dict()
        if not gaze_df.empty else {}
    )

    return {
        "n_trials": int(len(df)),
        "n_participants": int(df["user_id"].nunique()),
        "emotion_distribution": emotion_dist,
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 3),
        "avg_attention_time": round(float(df["attention_time"].mean()), 3),
        "engagement_by_stimulus": engagement_by_stimulus,
        "attention_by_stimulus": attention_by_stimulus,
    }


def segment_participants(df: pd.DataFrame) -> dict:
    """
    Aggregates per-user features and runs KMeans to produce consumer
    segments (personas) based on emotion/attention/engagement patterns.
    """
    user_features = df.groupby("user_id").agg(
        avg_emotion_score=("emotion_score", "mean"),
        avg_attention_time=("attention_time", "mean"),
        avg_engagement_score=("engagement_score", "mean"),
    ).reset_index()

    if len(user_features) < N_CLUSTERS:
        return {"warning": "Not enough unique participants for clustering yet.",
                "n_participants": len(user_features)}

    X = user_features[["avg_emotion_score", "avg_attention_time", "avg_engagement_score"]]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    user_features["segment"] = km.fit_predict(X_scaled)

    segment_profiles = (
        user_features.groupby("segment")[
            ["avg_emotion_score", "avg_attention_time", "avg_engagement_score"]
        ].mean().round(3).to_dict(orient="index")
    )

    return {
        "n_clusters": N_CLUSTERS,
        "segment_profiles": segment_profiles,
        "segment_sizes": user_features["segment"].value_counts().to_dict(),
    }


def run():
    df = run_preprocessing()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    summary = {
        "descriptive_stats": descriptive_stats(df),
        "segmentation": segment_participants(df),
    }

    out_path = os.path.join(REPORTS_DIR, "analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete. Saved to {out_path}")
    return summary


if __name__ == "__main__":
    run()