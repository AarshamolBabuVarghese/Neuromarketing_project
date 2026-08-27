"""
main.py
--------
Runs the full pipeline end-to-end AFTER data collection:
  preprocessing -> analysis -> A/B testing -> client report

Data collection itself is interactive (webcam + participant present),
so it's run separately:
  python -m src.experiment_controller --user_id 5

Full pipeline:
  python main.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import run as preprocess
from analysis import run as analyze
from ab_testing import run as ab_test
from generate_client_report import run as generate_report


def main():
    print("=== Step 1: Preprocessing ===")
    preprocess()

    print("\n=== Step 2: Analysis (descriptive stats + segmentation) ===")
    analyze()

    print("\n=== Step 3: A/B Testing ===")
    ab_test()

    print("\n=== Step 4: Client Report ===")
    path = generate_report(client_name="Client")
    print(f"\nDone. Final deliverable: {path}")


if __name__ == "__main__":
    main()