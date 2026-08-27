"""
Food Category Neuromarketing Experiment Controller
====================================================
Flow:
  1. Participant picks a food category via on-screen buttons (Tkinter).
  2. For that category, 5-7 stimuli are shown one at a time (OpenCV window).
  3. Each stimulus is shown for a randomized 5-6 sec window.
  4. During display, webcam frames are analyzed each tick to check whether
     the participant's face is detected (proxy for "looking at the screen").
     If attention drops below CONCENTRATION_THRESHOLD, an on-screen alert
     fires ("Please focus on the screen") and is logged.
  5. Participant rates the stimulus (keys 1-5 = purchase intent / liking)
     at any point during the window. Reaction time (speed) is measured
     from stimulus onset to keypress. If no key is pressed before the
     window closes, the trial is logged as "no_response" (invalid trial).
  6. All trial data is saved to a CSV per session (category, stimulus,
     feature tag, rating, reaction_time_ms, concentration_ratio,
     alert_triggered, valid_response).

Dependencies:
  pip install opencv-python

Folder structure expected (create these locally and drop your images in):
  stimuli/
    chips_namkeen/   -> 7 images
    beverages/       -> 6 images
    chocolates/      -> 5 images
    biscuits/        -> 6 images
    instant_food/    -> 7 images

Run:
  python experiment_controller_food.py
"""

import cv2
import os
import csv
import time
import random
import tkinter as tk
from datetime import datetime

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

STIMULI_ROOT = "stimuli"          # base folder holding category subfolders
RESULTS_DIR = "results"           # where CSV logs are saved
MIN_DISPLAY_SEC = 5.0              # minimum time per stimulus
MAX_DISPLAY_SEC = 6.0              # maximum time per stimulus
CONCENTRATION_THRESHOLD = 0.6      # min fraction of frames w/ face detected
ALERT_COOLDOWN_SEC = 1.5           # don't spam the alert every frame
WEBCAM_INDEX = 0

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# category_key -> (display_name, subfolder, [(filename, feature_tag), ...])
CATEGORY_CONFIG = {
    "1": {
        "name": "Chips & Namkeen",
        "folder": "chips_namkeen",
        "stimuli": [
            ("lays_classic.png", "packaging_color"),
            ("lays_spicy.png", "spice_level_claim"),
            ("bingo_mad_angles.png", "brand_logo_prominence"),
            ("haldiram_namkeen.png", "price_visibility"),
            ("kurkure.png", "transparency_window"),
            ("balaji_wafers.png", "regional_brand_trust"),
            ("too_yumm.png", "health_claim"),
        ],
    },
    "2": {
        "name": "Beverages",
        "folder": "beverages",
        "stimuli": [
            ("coca_cola.png", "bottle_shape"),
            ("pepsi.png", "color_of_liquid"),
            ("sprite.png", "condensation_freshness_cue"),
            ("real_juice.png", "celebrity_endorsement"),
            ("tropicana.png", "sugar_free_label"),
            ("thumsup.png", "brand_heritage_cue"),
        ],
    },
    "3": {
        "name": "Chocolates & Confectionery",
        "folder": "chocolates",
        "stimuli": [
            ("dairy_milk.png", "wrapper_color"),
            ("kitkat.png", "premium_texture_cue"),
            ("five_star.png", "size_perception"),
            ("munch.png", "price_value_combo"),
            ("perk.png", "gifting_cue"),
        ],
    },
    "4": {
        "name": "Biscuits & Cookies",
        "folder": "biscuits",
        "stimuli": [
            ("parle_g.png", "health_claim_digestive"),
            ("oreo.png", "brand_mascot"),
            ("good_day.png", "texture_visual"),
            ("hide_seek.png", "packaging_transparency"),
            ("marie_gold.png", "price_pack_size"),
            ("britannia_nutrichoice.png", "protein_health_tag"),
        ],
    },
    "5": {
        "name": "Instant / Ready-to-Eat",
        "folder": "instant_food",
        "stimuli": [
            ("maggi.png", "cooking_time_claim"),
            ("yippee.png", "spice_masala_visual"),
            ("top_ramen.png", "bowl_presentation"),
            ("knorr_soup.png", "health_tag"),
            ("mtr_readytoeat.png", "price_visibility"),
            ("ching_secret.png", "brand_logo_prominence"),
            ("wai_wai.png", "regional_brand_trust"),
        ],
    },
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# CATEGORY SELECTION UI (Tkinter buttons)
# ----------------------------------------------------------------------

def selection_gui():
    """Shows category choice buttons. Returns the chosen category_key or None."""
    chosen = {"key": None}

    root = tk.Tk()
    root.title("Neuromarketing Study — Choose a Food Category")
    root.geometry("420x360")

    tk.Label(
        root, text="Which category would you like to be tested on?",
        font=("Segoe UI", 13, "bold"), wraplength=380, pady=15
    ).pack()

    def pick(key):
        chosen["key"] = key
        root.destroy()

    for key, cfg in CATEGORY_CONFIG.items():
        tk.Button(
            root, text=cfg["name"], font=("Segoe UI", 11),
            width=28, height=2, command=lambda k=key: pick(k)
        ).pack(pady=6)

    root.mainloop()
    return chosen["key"]


# ----------------------------------------------------------------------
# ATTENTION / CONCENTRATION CHECK
# ----------------------------------------------------------------------

def face_present(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


# ----------------------------------------------------------------------
# SINGLE STIMULUS TRIAL
# ----------------------------------------------------------------------

def run_trial(cap, stim_path, feature_tag, category_name):
    """
    Displays one stimulus, runs the timed window, tracks concentration,
    captures rating + reaction time. Returns a dict of trial results.
    """
    img = cv2.imread(stim_path)
    if img is None:
        print(f"[WARN] Missing image, skipping: {stim_path}")
        return None

    display_duration = random.uniform(MIN_DISPLAY_SEC, MAX_DISPLAY_SEC)
    window_name = "Stimulus"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    start_time = time.time()
    rating = None
    reaction_time_ms = None
    frame_count = 0
    attentive_count = 0
    alert_triggered = False
    last_alert_time = 0

    while True:
        elapsed = time.time() - start_time
        remaining = display_duration - elapsed
        if remaining <= 0:
            break

        ok, cam_frame = cap.read()
        if ok:
            frame_count += 1
            attentive = face_present(cam_frame)
            if attentive:
                attentive_count += 1

            # Alert logic: if attention has dropped and cooldown has passed
            running_ratio = attentive_count / frame_count
            if (frame_count > 10 and running_ratio < CONCENTRATION_THRESHOLD
                    and time.time() - last_alert_time > ALERT_COOLDOWN_SEC):
                alert_triggered = True
                last_alert_time = time.time()
                print(f"[ALERT] Low concentration detected on '{feature_tag}' "
                      f"stimulus (ratio={running_ratio:.2f})")

        # Build display frame with overlay
        display_frame = img.copy()
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, f"{category_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(display_frame, f"Time left: {remaining:0.1f}s", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, "Rate 1(low)-5(high) purchase interest",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if alert_triggered and time.time() - last_alert_time < 1.0:
            cv2.putText(display_frame, "PLEASE FOCUS ON THE SCREEN",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if rating is None and key in [ord(str(n)) for n in range(1, 6)]:
            rating = int(chr(key))
            reaction_time_ms = round((time.time() - start_time) * 1000, 1)

        if key == 27:  # ESC to abort whole session
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Session aborted by participant.")

    cv2.destroyWindow(window_name)

    concentration_ratio = round(attentive_count / frame_count, 3) if frame_count else 0.0
    valid_response = rating is not None

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category_name,
        "stimulus": os.path.basename(stim_path),
        "feature_tag": feature_tag,
        "rating_1to5": rating if rating is not None else "",
        "reaction_time_ms": reaction_time_ms if reaction_time_ms is not None else "",
        "concentration_ratio": concentration_ratio,
        "alert_triggered": alert_triggered,
        "valid_response": valid_response,
        "display_duration_sec": round(display_duration, 2),
    }


# ----------------------------------------------------------------------
# RUN FULL EXPERIMENT FOR A CATEGORY
# ----------------------------------------------------------------------

def run_experiment(category_key):
    cfg = CATEGORY_CONFIG[category_key]
    category_name = cfg["name"]
    folder = os.path.join(STIMULI_ROOT, cfg["folder"])

    stimuli_order = cfg["stimuli"][:]
    random.shuffle(stimuli_order)  # randomize presentation order per participant

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Check camera permissions/index.")
        return

    results = []
    try:
        for filename, feature_tag in stimuli_order:
            stim_path = os.path.join(folder, filename)
            trial_result = run_trial(cap, stim_path, feature_tag, category_name)
            if trial_result:
                results.append(trial_result)
                print(f"  -> {filename} | rating={trial_result['rating_1to5']} "
                      f"| RT={trial_result['reaction_time_ms']}ms "
                      f"| concentration={trial_result['concentration_ratio']}")
    except KeyboardInterrupt as e:
        print(str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if results:
        save_results(category_name, results)
        print_summary(results)
    else:
        print("No trials completed — nothing saved.")


# ----------------------------------------------------------------------
# STORAGE
# ----------------------------------------------------------------------

def save_results(category_name, results):
    fname = f"{category_name.replace(' ', '_').replace('&', 'and')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(RESULTS_DIR, fname)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] Results written to {path}")


# ----------------------------------------------------------------------
# QUICK SUMMARY (feeds into your existing analysis.py / dashboard.py)
# ----------------------------------------------------------------------

def print_summary(results):
    valid = [r for r in results if r["valid_response"]]
    n = len(results)
    n_valid = len(valid)
    avg_rt = (sum(r["reaction_time_ms"] for r in valid) / n_valid) if n_valid else None
    avg_concentration = sum(r["concentration_ratio"] for r in results) / n
    n_alerts = sum(1 for r in results if r["alert_triggered"])

    print("\n----- SESSION SUMMARY -----")
    print(f"Trials completed:        {n}")
    print(f"Valid responses:         {n_valid} ({n_valid/n*100:.0f}% response accuracy)")
    print(f"Avg reaction time:       {avg_rt:.0f} ms" if avg_rt else "Avg reaction time: N/A")
    print(f"Avg concentration ratio: {avg_concentration:.2f}")
    print(f"Attention alerts fired:  {n_alerts}")
    print("----------------------------\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    category_key = selection_gui()
    if category_key is None:
        print("No category selected. Exiting.")
    else:
        print(f"Starting experiment for: {CATEGORY_CONFIG[category_key]['name']}")
        run_experiment(category_key)