"""
ab_testing.py
--------------
Within-subject A/B(/C) creative testing: is the engagement/emotion
difference between stimulus variants (e.g. "A (Left)" vs "B (Right)")
statistically significant, or just noise?

Uses Mann-Whitney U (non-parametric - safer than a t-test for small,
possibly non-normal samples like early-stage pilot data).
"""
import json
import os
import yaml
import pandas as pd
from scipy.stats import mannwhitneyu

from preprocessing import run as run_preprocessing

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

REPORTS_DIR = _CFG["paths"]["reports_dir"]
ALPHA = _CFG["analysis"]["significance_level"]


def compare_variants(df: pd.DataFrame, metric="engagement_score"):
    """
    Runs pairwise Mann-Whitney U tests between every pair of stimulus
    variants found in the 'Eye direction' column, on the given metric.
    """
    gaze_df = df[df["usable_for_gaze_analysis"]]
    variants = sorted(gaze_df["Eye direction"].dropna().unique())

    results = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a_name, b_name = variants[i], variants[j]
            a = gaze_df.loc[gaze_df["Eye direction"] == a_name, metric].dropna()
            b = gaze_df.loc[gaze_df["Eye direction"] == b_name, metric].dropna()

            if len(a) < 3 or len(b) < 3:
                results.append({
                    "variant_a": a_name, "variant_b": b_name,
                    "note": "insufficient samples (need >= 3 per variant)",
                    "n_a": len(a), "n_b": len(b),
                })
                continue

            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
            winner = a_name if a.mean() > b.mean() else b_name

            results.append({
                "variant_a": a_name, "variant_b": b_name,
                "mean_a": round(float(a.mean()), 3),
                "mean_b": round(float(b.mean()), 3),
                "p_value": round(float(p_value), 4),
                "significant": bool(p_value < ALPHA),
                "likely_better_variant": winner if p_value < ALPHA else "no significant difference",
                "n_a": int(len(a)), "n_b": int(len(b)),
            })
    return results


def run():
    df = run_preprocessing()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    ab_results = {
        "engagement_score": compare_variants(df, "engagement_score"),
        "emotion_score": compare_variants(df, "emotion_score"),
        "attention_time": compare_variants(df, "attention_time"),
    }

    out_path = os.path.join(REPORTS_DIR, "ab_test_results.json")
    with open(out_path, "w") as f:
        json.dump(ab_results, f, indent=2)

    print(f"A/B test results saved to {out_path}")
    return ab_results


if __name__ == "__main__":
    run()