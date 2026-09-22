"""
experiment_controller.py
---------------------------------------------------------------
Runs a full A/B composite-image session: loops through every
stimulus listed in stimulus_manifest.csv, calls capture_pipeline
.run_trial() for each one (fullscreen image + webcam sampling),
then writes the result into unified_dataset_clean.csv via
storage.append_trial() -- filling in BOTH 'Product Features' and
'Eye direction'.

Before the session starts, the participant is shown an
Instamart-style category picker (Namkeen / Chips / Beverages /
etc., pulled from the manifest's 'category' column) and can choose
to be tested on just ONE category instead of always running the
full stimulus set.

FIX (this version): capture_pipeline.py now opens the camera and
fullscreen window ONCE per session instead of once per trial (see
its docstring for why). This file is updated to match: it calls
open_session() once before the loop, passes the same `cap` into
every run_trial() call, and calls close_session() once after the
loop -- even if a trial errors out, via try/finally.

Put your composite images (like your neuinsta A|B picture) in:
  stimuli/ab_composites/

Drop this file into: src/experiment_controller.py
Run:
  python src/experiment_controller.py --user_id 101
---------------------------------------------------------------
"""
import argparse
import os
import csv

from capture_pipeline import run_trial, open_session, close_session
import storage

MANIFEST_PATH = os.path.join("data", "stimulus_manifest.csv")
STIMULI_DIR = os.path.join("stimuli", "ab_composites")


def load_manifest(path=MANIFEST_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found at {path}. Copy stimulus_manifest.csv "
            f"there and fill in your composite image filenames."
        )
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_category(manifest):
    """
    Instamart-style category picker: shown once per session, before
    any stimuli. Lets the participant choose ONE category to be
    tested on instead of always running the full stimulus set.
    """
    categories = sorted({row["category"] for row in manifest if row.get("category")})
    if not categories:
        return manifest  # manifest has no category column — show everything

    print("\nWhat would you like to test today?\n")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat.title()}")
    print(f"  {len(categories) + 1}. All categories")

    choice = input("\nEnter a number: ").strip()
    try:
        idx = int(choice)
        if idx == len(categories) + 1:
            return manifest
        if 1 <= idx <= len(categories):
            selected = categories[idx - 1]
            filtered = [row for row in manifest if row.get("category") == selected]
            print(f"\nShowing {len(filtered)} stimuli in '{selected}'.\n")
            return filtered
    except ValueError:
        pass

    print("Invalid choice — showing all categories instead.\n")
    return manifest


def run_session(user_id):
    manifest = load_manifest()
    manifest = select_category(manifest)
    print(f"Starting A/B session for user_id={user_id} — {len(manifest)} stimuli to show.\n")

    # Camera + fullscreen window open ONCE for the whole session.
    cap = open_session()

    try:
        for row in manifest:
            img_path = os.path.join(STIMULI_DIR, row["image_filename"])
            product_feature_label = f"{row['product_name']}_{row['attribute_changed']}"

            print(f"Showing: {row['image_filename']}  ->  feature test: {product_feature_label}")

            try:
                result = run_trial(user_id=user_id, cap=cap, stimulus_path=img_path)
            except RuntimeError as e:
                print(f"  [skip] {e}")
                continue

            saved_row = storage.append_trial(
                user_id=user_id,
                emotion=result["emotion"],
                gaze_x=result["gaze_x"],
                gaze_y=result["gaze_y"],
                attention_time=result["attention_time"],
                emotion_score=result["emotion_score"],
                product_order=product_feature_label,   # -> "Product Features" column
                aoi_label=aoi_to_variant(result["aoi"]),  # -> "Eye direction" column
            )

            print(f"  -> emotion={saved_row['emotion']}, "
                  f"Eye direction={saved_row['Eye direction']}, "
                  f"engagement_score={saved_row['engagement_score']}\n")
    finally:
        # Camera + window close ONCE, after every stimulus is done --
        # runs even if a trial raised an error mid-session.
        close_session(cap)

    storage.log_session_meta(user_id, note=f"A/B session, {len(manifest)} stimuli")
    print("Session complete. Data written to unified_dataset_clean.csv")


def aoi_to_variant(aoi_label):
    """
    Your config.yaml AOI keys are 'A (Left)', 'B (Right)', 'C (Neutral)'.
    ab_recommendation_engine.py expects plain 'A' / 'B' / 'C'.
    """
    if not aoi_label:
        return "NA"
    return aoi_label.split(" ")[0]  # "A (Left)" -> "A"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True, help="Participant ID (any unique string/number)")
    args = parser.parse_args()
    run_session(args.user_id)