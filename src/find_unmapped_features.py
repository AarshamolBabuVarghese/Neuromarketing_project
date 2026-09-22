"""
find_unmapped_features.py
---------------------------------------------------------------
Diagnostic for the "UNKNOWN (UNKNOWN)" block in your report: lists
every distinct 'Product Features' value actually present in
unified_dataset_clean.csv that does NOT match any test_name (or
product_name_attribute_changed combo) in stimulus_manifest.csv.

These are either: (a) stale data from an old naming scheme before
you added product_id/category to the manifest, or (b) a real typo
mismatch between what experiment_controller.py wrote and what's in
the manifest. Either way, they're currently getting silently lumped
into one fake "UNKNOWN" product in the report.

Run:
  python src/find_unmapped_features.py
---------------------------------------------------------------
"""
import pandas as pd
from rec import DATASET_PATH, load_manifest_lookup, load_stimulus_subset, FEATURE_COL

df = pd.read_csv(DATASET_PATH)
stim_df = load_stimulus_subset(df)
lookup = load_manifest_lookup()

value_counts = stim_df[FEATURE_COL].value_counts()

print(f"Total distinct 'Product Features' values in dataset: {len(value_counts)}")
print(f"Total keys registered in manifest lookup: {len(lookup)}\n")

unmapped = [v for v in value_counts.index if v not in lookup]

if not unmapped:
    print("Everything matches. No unmapped values found.")
else:
    print(f"UNMAPPED values ({len(unmapped)}) — these are what's collapsing into 'UNKNOWN':\n")
    for v in unmapped:
        print(f"  '{v}'  ({value_counts[v]} rows)")
    print("\nFix options:")
    print("  1. If these are old/test data from before the manifest existed -- "
          "delete those rows from unified_dataset_clean.csv, or")
    print("  2. If these are real tests you want to keep -- add a matching row "
          "to stimulus_manifest.csv with this exact value as the test_name.")