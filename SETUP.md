# Neuromarketing Website — Setup Guide

Two brand-new folders. Nothing in your existing `src/`, `data/`,
`results/`, `reports/`, `main.py` is touched or imported in a way that
changes their behavior.

## 1. Where to put things

Unzip so you end up with this, sitting next to your existing folders:

```
Neuromarketing_project/
├── src/                  ← already exists, untouched
├── data/                 ← already exists, untouched
├── results/              ← already exists, untouched
├── reports/              ← already exists, untouched
├── main.py               ← already exists, untouched
├── config/
├── webapp_backend/       ← NEW — copy this in
│   ├── app.py
│   └── requirements.txt
└── webapp_frontend/      ← NEW — copy this in
    ├── package.json
    ├── vite.config.js
    ├── index.html
    ├── .env
    └── src/...
```

## 2. Install the backend's one extra dependency

In Terminal, from your project root, using your **existing** venv
(the one that already has Flask, pandas, streamlit installed):

```bash
source .venv/bin/activate      # or however you activate your venv
pip install -r webapp_backend/requirements.txt
```

That installs only `PyJWT` — everything else (Flask, flask-cors,
pandas, streamlit) is already in your project's requirements.txt.

## 3. Install the frontend dependencies

You need Node.js installed on your Mac (get it from nodejs.org if you
don't have it — check with `node -v` in Terminal).

```bash
cd webapp_frontend
npm install
```

## 4. Run everything (3 terminals)

**Terminal 1 — your existing venv, backend API:**
```bash
cd Neuromarketing_project
source .venv/bin/activate
python webapp_backend/app.py
```
Runs on http://127.0.0.1:5000 — this is a brand new file, does not
run main.py or dashboard.py itself, just knows how to *launch* them.

**Terminal 2 — frontend:**
```bash
cd Neuromarketing_project/webapp_frontend
npm run dev
```
Runs on http://localhost:5173 — open this in your browser.

**You don't need a 3rd terminal for Streamlit** — clicking
"Dashboard" in the website tells the backend to run
`streamlit run src/dashboard.py` for you automatically, exactly the
command you already use.

## 5. First use

1. Open http://localhost:5173 → you'll land on **Register**.
2. Create a username/password (stored in a new local file,
   `webapp_backend/users.db` — separate from your project data).
3. You'll land on the Home screen with 6 option cards:
   - **Collect Data** → runs a live A/B session in the browser:
     enter a participant ID, do the 5-point red-dot calibration,
     pick a category, then step through stimuli with your webcam —
     same logic as `experiment_controller.py`, just browser-driven.
     Your browser will ask for camera permission the first time.
   - **Run Analysis** → runs `main.py` on already-collected data,
     streams the log live.
   - **Dataset** → paginated view of `unified_dataset_clean.csv`.
   - **Dashboard** → boots `streamlit run src/dashboard.py` and shows
     it in an iframe, right inside the site.
   - **AI Recommendations** → reads `rec.py`'s existing output CSVs.
   - **Reports** → lists and downloads files from `reports/`.

### Notes on Collect Data specifically
- Needs your Mac's camera permission for the browser (Chrome/Safari
  will prompt on first use — allow it).
- Only run one session at a time — the backend tracks one active
  trial/calibration in memory, same one-session-at-a-time design as
  `experiment_controller.py`.
- If you refresh the page mid-session, start over from Collect Data —
  in-progress calibration/trial state is not preserved across a
  page reload.
- Results are written into `unified_dataset_clean.csv` and
  `data/processed_sessions/calibration_<id>.json` exactly the way
  the terminal version does, so `rec.py`, `main.py`, and
  `dashboard.py` all keep working on this data unchanged.

## Notes

- If port 5000 or 5173 is already taken on your Mac, change the port
  in `webapp_backend/app.py` (`app.run(..., port=5000)`) or
  `webapp_frontend/vite.config.js` (`server.port`), and update
  `webapp_frontend/.env`'s `VITE_API_URL` to match.
- "Run Analysis" blocks a second run from starting while one is in
  progress, since `main.py` isn't designed for concurrent runs.
- The website never edits your CSVs, `main.py`, or anything in `src/`
  — it only reads existing output files and launches your existing
  scripts as subprocesses, the same way you already run them.
