"""
preprocessing.py
------------------
Cleans unified_dataset_clean.csv before it goes into analysis.py.

Handles the exact issues you already found in the data:
  1. Baseline rows (no real product-testing session yet) have
     "Product Features"/"Eye direction" = NA -> these are kept for
     emotion-only baseline stats, but excluded from AOI/gaze analysis.
  2. Placeholder/repeating gaze coordinates (e.g. the same gaze_x for
     an entire session) indicate the gaze tracker didn't get a real
     read -> flagged and excluded from spatial analysis, kept for
     emotion analysis.
  3. Type coercion + duplicate trial_id removal.
"""
import os
import pandas as pd
import yaml

with open("config/config.yaml", "r") as f:
    _CFG = yaml.safe_load(f)

UNIFIED_PATH = _CFG["paths"]["unified_dataset"]
PROCESSED_DIR = _CFG["paths"]["processed_dir"]


def load_raw():
    df = pd.read_csv(UNIFIED_PATH)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Type coercion
    numeric_cols = ["gaze_x", "gaze_y", "attention_time", "emotion_score", "engagement_score"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2. Drop exact duplicate trials
    df = df.drop_duplicates(subset=["trial_id"], keep="first")

    # 3. Flag rows with no real product-testing data (baseline-only rows)
    df["has_product_data"] = df["Product Features"].notna() & (df["Product Features"] != "NA")

    # 4. Flag placeholder / non-varying gaze within the same user+session
    #    (repeated identical (gaze_x, gaze_y) across consecutive trials for a user
    #     strongly suggests the tracker wasn't actually reading new positions)
    df = df.sort_values(["user_id", "trial_id"])
    dup_gaze = (
        df.groupby("user_id")[["gaze_x", "gaze_y"]]
        .apply(lambda g: g.duplicated(keep=False))
        .reset_index(level=0, drop=True)
    )
    df["gaze_is_placeholder"] = dup_gaze.fillna(False)

    # 5. Rows usable for spatial/AOI analysis: real product data + non-placeholder gaze
    df["usable_for_gaze_analysis"] = df["has_product_data"] & (~df["gaze_is_placeholder"])

    # 6. Drop rows with missing core fields
    df = df.dropna(subset=["emotion", "emotion_score", "attention_time", "engagement_score"])

    return df.reset_index(drop=True)


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_raw()
    clean_df = clean(df)

    out_path = os.path.join(PROCESSED_DIR, "unified_dataset_processed.csv")
    clean_df.to_csv(out_path, index=False)

    print(f"Loaded {len(df)} raw rows -> {len(clean_df)} clean rows")
    print(f"Rows usable for gaze/AOI analysis: {clean_df['usable_for_gaze_analysis'].sum()}")
    print(f"Saved to {out_path}")
    return clean_df


if __name__ == "__main__":
    run()