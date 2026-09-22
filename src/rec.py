"""
ab_recommendation_engine.py
---------------------------------------------------------------
Groups by 'Product Features' (one specific attribute test per
group), runs A/B significance tests within each group, and tags
every result with product_id/product_name/category/attribute_changed
from the manifest.

v3 CHANGE: a test with too little data used to just say "no clear
winner" with zero other information -- useless when you're still
early in data collection and EVERY test looks like that. Now every
result also carries a directional 'leaning_variant'/'leaning_value'
(whichever variant currently averages higher, even if not yet
significant) plus a rough 'collect N more responses' estimate, so
the report can show real signal-in-progress instead of a flat
placeholder sentence repeated for every product.

Drop into: Neuromarketing_project/src/ab_recommendation_engine.py
---------------------------------------------------------------
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, asdict
from typing import List, Optional

DATASET_PATH = os.path.join("data", "unified_dataset_clean.csv")
MANIFEST_PATH = os.path.join("data", "stimulus_manifest.csv")
OUT_DIR = os.path.join("results", "ab_recommendations")

METRIC_COLS = ["engagement_score", "emotion_score", "attention_time"]
FEATURE_COL = "Product Features"
VARIANT_COL = "Eye direction"

MANIFEST_OPTIONAL_COLS = ["product_id", "product_name", "category", "attribute_changed"]

# Rough rule of thumb for "how many total responses before this kind
# of test is typically well-powered" -- not a rigorous power
# calculation, just a practical target so the report can say
# something more useful than "collect more data". Overridable via
# --target-n for quick pipeline testing with small numbers.
TARGET_N_PER_VARIANT = 20

# Minimum samples per variant before a significance test even runs.
# Overridable via --min-n for quick pipeline testing (e.g. --min-n 2
# lets you confirm ADOPT/CONSIDER logic fires correctly with a
# handful of test participants, before running the real study at the
# real threshold of 3+).
MIN_SAMPLES_PER_VARIANT = 3


def load_manifest_lookup(path: str = MANIFEST_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    manifest = pd.read_csv(path)
    lookup = {}
    for _, row in manifest.iterrows():
        entry = {"A": row["product_a_name"], "B": row["product_b_name"]}
        for col in MANIFEST_OPTIONAL_COLS:
            entry[col] = row[col] if col in manifest.columns and pd.notna(row.get(col)) else "UNKNOWN"

        candidate_keys = {row["test_name"]}
        if "product_name" in manifest.columns and "attribute_changed" in manifest.columns \
                and pd.notna(row.get("product_name")) and pd.notna(row.get("attribute_changed")):
            candidate_keys.add(f"{row['product_name']}_{row['attribute_changed']}")
        for key in candidate_keys:
            lookup[key] = entry
    return lookup


@dataclass
class FeatureTestResult:
    product_feature: str
    winner_variant: str
    winner_value: Optional[str]
    n_a: int
    n_b: int
    metric_results: dict
    composite_confidence: str
    recommendation: str
    product_id: str = "UNKNOWN"
    product_name: str = "UNKNOWN"
    category: str = "UNKNOWN"
    attribute_changed: str = "UNKNOWN"
    leaning_variant: Optional[str] = None
    leaning_value: Optional[str] = None
    n_needed_estimate: Optional[int] = None


def load_stimulus_subset(df: pd.DataFrame) -> pd.DataFrame:
    if FEATURE_COL not in df.columns or VARIANT_COL not in df.columns:
        raise ValueError(f"Expected columns '{FEATURE_COL}' and '{VARIANT_COL}' not found. "
                          f"Found: {list(df.columns)}")
    subset = df[df[FEATURE_COL].notna() & df[VARIANT_COL].notna()].copy()
    subset[VARIANT_COL] = (
        subset[VARIANT_COL].astype(str).str.strip().str.split().str[0].str.upper()
    )
    return subset


def compare_variants_within_feature(group: pd.DataFrame, product_feature: str,
                                     manifest_entry: Optional[dict] = None) -> FeatureTestResult:
    variants = sorted(group[VARIANT_COL].dropna().unique().tolist())
    metric_results = {}
    sig_count = 0
    winner_votes = {}
    lean_votes = {}  # tally of which variant averages higher per metric, regardless of significance

    identity_kwargs = {
        "product_id": (manifest_entry or {}).get("product_id", "UNKNOWN"),
        "product_name": (manifest_entry or {}).get("product_name", "UNKNOWN"),
        "category": (manifest_entry or {}).get("category", "UNKNOWN"),
        "attribute_changed": (manifest_entry or {}).get("attribute_changed", "UNKNOWN"),
    }

    if "A" not in variants or "B" not in variants:
        return FeatureTestResult(
            product_feature=product_feature, winner_variant="No clear winner", winner_value=None,
            n_a=0, n_b=0, metric_results={}, composite_confidence="Low",
            recommendation=f"'{product_feature}': only one variant has data so far — need both A and B represented.",
            **identity_kwargs,
        )

    group_a = group[group[VARIANT_COL] == "A"]
    group_b = group[group[VARIANT_COL] == "B"]
    n_a, n_b = len(group_a), len(group_b)

    for metric in METRIC_COLS:
        if metric not in group.columns:
            continue
        a_vals = group_a[metric].dropna()
        b_vals = group_b[metric].dropna()

        if len(a_vals) == 0 or len(b_vals) == 0:
            continue

        mean_a, mean_b = float(a_vals.mean()), float(b_vals.mean())
        if mean_a != mean_b:
            lean_votes[("A" if mean_a > mean_b else "B")] = lean_votes.get(
                ("A" if mean_a > mean_b else "B"), 0) + 1

        if len(a_vals) < MIN_SAMPLES_PER_VARIANT or len(b_vals) < MIN_SAMPLES_PER_VARIANT:
            metric_results[metric] = {
                "mean_a": round(mean_a, 3), "mean_b": round(mean_b, 3),
                "note": f"insufficient samples (n_a={len(a_vals)}, n_b={len(b_vals)}, need >={MIN_SAMPLES_PER_VARIANT} each)",
            }
            continue

        if a_vals.std() == 0 and b_vals.std() == 0:
            metric_results[metric] = {
                "mean_a": round(mean_a, 3), "mean_b": round(mean_b, 3),
                "note": "no variation in either group",
            }
            continue

        tstat, p_value = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        significant = bool(p_value < 0.05)
        better = ("A" if mean_a > mean_b else "B") if significant else None
        if significant:
            sig_count += 1
            winner_votes[better] = winner_votes.get(better, 0) + 1

        metric_results[metric] = {
            "mean_a": round(mean_a, 3), "mean_b": round(mean_b, 3),
            "p_value": round(float(p_value), 4), "significant": significant,
            "better_variant": better,
        }

    winner_variant = max(winner_votes, key=winner_votes.get) if winner_votes else "No clear winner"
    confidence = "High" if sig_count >= 2 else "Medium" if sig_count == 1 else "Low"

    # Directional lean, shown even when not statistically proven yet.
    leaning_variant = max(lean_votes, key=lean_votes.get) if lean_votes else None
    leaning_value = (manifest_entry or {}).get(leaning_variant) if leaning_variant else None

    n_needed_estimate = None
    if winner_variant == "No clear winner":
        shortfall = max(TARGET_N_PER_VARIANT - n_a, TARGET_N_PER_VARIANT - n_b, 0)
        n_needed_estimate = shortfall if shortfall > 0 else None

    winner_value = (manifest_entry or {}).get(winner_variant) if winner_variant in ("A", "B") else None
    recommendation = build_recommendation(
        product_feature, winner_variant, winner_value, metric_results, confidence,
        leaning_variant, leaning_value, n_a, n_b, n_needed_estimate,
    )

    return FeatureTestResult(
        product_feature=product_feature, winner_variant=winner_variant, winner_value=winner_value,
        n_a=n_a, n_b=n_b, metric_results=metric_results, composite_confidence=confidence,
        recommendation=recommendation, leaning_variant=leaning_variant, leaning_value=leaning_value,
        n_needed_estimate=n_needed_estimate,
        **identity_kwargs,
    )


def build_recommendation(product_feature, winner_variant, winner_value, metric_results,
                          confidence, leaning_variant, leaning_value, n_a, n_b,
                          n_needed_estimate) -> str:
    if winner_variant != "No clear winner":
        sig_metrics = [f"{m.replace('_', ' ')} ({r['mean_a']} vs {r['mean_b']}, p={r['p_value']})"
                       for m, r in metric_results.items() if r.get("significant")]
        detail = "; ".join(sig_metrics) if sig_metrics else "trend not statistically significant yet"
        action = f"'{winner_value}' outperformed the alternative" if winner_value else f"Variant {winner_variant} outperformed"
        return (f"For '{product_feature}' — {action}. "
                f"{confidence} confidence — significant on: {detail}.")

    # Not statistically proven, but still say what the data leans toward.
    base = f"'{product_feature}': not statistically significant yet (n_a={n_a}, n_b={n_b})."
    if leaning_variant and leaning_value:
        base += f" Currently trending toward '{leaning_value}' — too early to act on."
    elif leaning_variant:
        base += f" Currently trending toward Variant {leaning_variant} — too early to act on."
    if n_needed_estimate:
        base += f" Collect roughly {n_needed_estimate} more participant(s) on this test to reach a reliable read."
    return base


def run_all_feature_tests(csv_path: str = DATASET_PATH, category: Optional[str] = None) -> List[FeatureTestResult]:
    df = pd.read_csv(csv_path)
    stim_df = load_stimulus_subset(df)
    manifest_lookup = load_manifest_lookup()

    results = []
    for product_feature, group in stim_df.groupby(FEATURE_COL):
        manifest_entry = manifest_lookup.get(product_feature)
        if category and (manifest_entry or {}).get("category", "").lower() != category.lower():
            continue
        results.append(compare_variants_within_feature(group, product_feature, manifest_entry))

    order = {"High": 0, "Medium": 1, "Low": 2}
    results.sort(key=lambda r: (r.winner_variant == "No clear winner", order[r.composite_confidence]))
    return results


def export_results(results: List[FeatureTestResult], out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ab_recommendations.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    flat = pd.DataFrame([{
        "product_id": r.product_id, "product_name": r.product_name, "category": r.category,
        "product_feature": r.product_feature, "attribute_changed": r.attribute_changed,
        "winner_variant": r.winner_variant, "winner_value": r.winner_value,
        "confidence": r.composite_confidence, "n_a": r.n_a, "n_b": r.n_b,
        "leaning_variant": r.leaning_variant, "leaning_value": r.leaning_value,
        "n_needed_estimate": r.n_needed_estimate,
        "recommendation": r.recommendation,
    } for r in results])
    flat.to_csv(os.path.join(out_dir, "ab_recommendations.csv"), index=False)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None,
                         help="Only analyze one category, e.g. 'biscuits'. Omit for all categories combined.")
    parser.add_argument("--min-n", type=int, default=None,
                         help=f"Override minimum samples per variant needed to run a significance test "
                              f"(default {MIN_SAMPLES_PER_VARIANT}). Lower this (e.g. 2) to quickly confirm "
                              f"the pipeline produces ADOPT/CONSIDER results with a small test batch of "
                              f"participants, before running the real study.")
    parser.add_argument("--target-n", type=int, default=None,
                         help=f"Override the 'collect N more participants' target used in the report "
                              f"(default {TARGET_N_PER_VARIANT}). Only affects the guidance text, not the "
                              f"significance test itself.")
    args = parser.parse_args()

    if args.min_n is not None:
        MIN_SAMPLES_PER_VARIANT = args.min_n
        print(f"[test mode] MIN_SAMPLES_PER_VARIANT overridden to {MIN_SAMPLES_PER_VARIANT}")
    if args.target_n is not None:
        TARGET_N_PER_VARIANT = args.target_n
        print(f"[test mode] TARGET_N_PER_VARIANT overridden to {TARGET_N_PER_VARIANT}")

    results = run_all_feature_tests(category=args.category)
    print(f"\nFound {len(results)} distinct product-feature tests"
          f"{f' in category {args.category}' if args.category else ' (all categories)'}.\n")
    for r in results:
        print(f"[{r.composite_confidence}] ({r.product_name}) {r.recommendation}")
    export_results(results)