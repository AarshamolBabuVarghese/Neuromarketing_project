"""
One-off fix: renames the 'user_id.1' column to 'trial_id' in
data/unified_dataset_clean.csv, in place. Run this once, then delete it.
"""
import pandas as pd

PATH = "data/unified_dataset_clean.csv"

df = pd.read_csv(PATH)
print("Columns before:", list(df.columns))

if "user_id.1" in df.columns:
    df = df.rename(columns={"user_id.1": "trial_id"})
    df.to_csv(PATH, index=False)
    print("Renamed 'user_id.1' -> 'trial_id' and saved.")
else:
    print("'user_id.1' not found - no change needed (column may already be correct).")

print("Columns after:", list(df.columns))