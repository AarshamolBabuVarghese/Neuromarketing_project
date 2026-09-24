"""
webapp_backend/app.py
---------------------------------------------------------------
Standalone Flask API for the Neuromarketing project's website.

IMPORTANT: this file only READS from your existing project
(data/, results/, reports/) and launches your existing scripts
(main.py, src/dashboard.py) as separate subprocesses. It never
imports, edits, or overwrites anything in src/.

Folder placement (sibling to your existing folders):
  Neuromarketing_project/
    src/
    data/
    results/
    reports/
    main.py
    webapp_backend/      <-- this file goes here
      app.py
      requirements.txt
      users.db            (auto-created on first run)

Run (from the Neuromarketing_project root, inside your existing venv):
  pip install -r webapp_backend/requirements.txt
  python webapp_backend/app.py

Runs on http://127.0.0.1:5000
---------------------------------------------------------------
"""
import os
import sqlite3
import socket
import subprocess
import sys
import uuid
import datetime
from functools import wraps

import jwt
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------------------------------------
# Paths — PROJECT_ROOT is the folder ABOVE webapp_backend/, i.e. your
# existing Neuromarketing_project root. Nothing here writes into src/.
# ------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATASET_PATH = os.path.join(DATA_DIR, "unified_dataset_clean.csv")
REC_CSV = os.path.join(RESULTS_DIR, "ab_recommendations", "ab_recommendations.csv")
PRODUCT_CSV = os.path.join(RESULTS_DIR, "ab_recommendations", "product_reports", "product_level_recommendations.csv")
DB_PATH = os.path.join(BACKEND_DIR, "users.db")
LOG_DIR = os.path.join(BACKEND_DIR, "job_logs")
os.makedirs(LOG_DIR, exist_ok=True)

SECRET_KEY = os.environ.get("WEBAPP_SECRET_KEY", "dev-secret-change-me")
STREAMLIT_PORT = 8501

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

# Browser-based data-collection endpoints (calibration + live trial
# capture) — registered from a separate new file, live_capture.py.
from live_capture import live_bp
app.register_blueprint(live_bp)

# ------------------------------------------------------------------
# DB setup (separate sqlite file, only for website logins)
# ------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------
def make_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/api/register", methods=["POST"])
def register():
    body = request.get_json(force=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "That username is already taken"}), 409
    finally:
        conn.close()

    return jsonify({"token": make_token(username), "username": username}), 201


@app.route("/api/login", methods=["POST"])
def login():
    body = request.get_json(force=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify({"token": make_token(username), "username": username}), 200


# ------------------------------------------------------------------
# Dataset (reads the same file dashboard.py reads — read only)
# ------------------------------------------------------------------
@app.route("/api/dataset", methods=["GET"])
@token_required
def get_dataset():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": f"Dataset not found at {DATASET_PATH}"}), 404

    df = pd.read_csv(DATASET_PATH)
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 50))
    start = (page - 1) * page_size
    end = start + page_size

    return jsonify({
        "total_rows": len(df),
        "columns": list(df.columns),
        "page": page,
        "page_size": page_size,
        "rows": df.iloc[start:end].fillna("").to_dict(orient="records"),
    })


# ------------------------------------------------------------------
# AI Recommendations (reads rec.py's already-generated output)
# ------------------------------------------------------------------
@app.route("/api/recommendations", methods=["GET"])
@token_required
def get_recommendations():
    if not os.path.exists(REC_CSV):
        return jsonify({"error": "No recommendations yet — run the analysis first."}), 404

    rec_df = pd.read_csv(REC_CSV)
    product_df = pd.read_csv(PRODUCT_CSV) if os.path.exists(PRODUCT_CSV) else pd.DataFrame()

    return jsonify({
        "recommendations": rec_df.fillna("").to_dict(orient="records"),
        "product_summary": product_df.fillna("").to_dict(orient="records"),
    })


# ------------------------------------------------------------------
# Reports (lists / serves files already written into reports/)
# ------------------------------------------------------------------
@app.route("/api/reports", methods=["GET"])
@token_required
def list_reports():
    if not os.path.exists(REPORTS_DIR):
        return jsonify({"files": []})
    files = []
    for name in os.listdir(REPORTS_DIR):
        full = os.path.join(REPORTS_DIR, name)
        if os.path.isfile(full):
            files.append({
                "name": name,
                "modified": datetime.datetime.fromtimestamp(os.path.getmtime(full)).isoformat(),
                "size_kb": round(os.path.getsize(full) / 1024, 1),
            })
    return jsonify({"files": sorted(files, key=lambda f: f["modified"], reverse=True)})


@app.route("/api/reports/download/<path:filename>", methods=["GET"])
@token_required
def download_report(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


# ------------------------------------------------------------------
# Run Analysis — launches your existing main.py as a subprocess.
# Does not import or modify main.py / src/*.py in any way.
# ------------------------------------------------------------------
JOBS = {}  # job_id -> {"process": Popen, "log_path": str}

@app.route("/api/analysis/run", methods=["POST"])
@token_required
def run_analysis():
    for job in JOBS.values():
        if job["process"].poll() is None:
            return jsonify({"error": "An analysis run is already in progress", "job_id": job["job_id"]}), 409

    job_id = str(uuid.uuid4())
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    JOBS[job_id] = {"process": process, "log_path": log_path, "job_id": job_id}
    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/api/analysis/status/<job_id>", methods=["GET"])
@token_required
def analysis_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job id"}), 404

    return_code = job["process"].poll()
    status = "running" if return_code is None else ("success" if return_code == 0 else "failed")

    log_tail = ""
    if os.path.exists(job["log_path"]):
        with open(job["log_path"], "r", errors="ignore") as f:
            lines = f.readlines()
            log_tail = "".join(lines[-60:])

    return jsonify({"status": status, "return_code": return_code, "log_tail": log_tail})


# ------------------------------------------------------------------
# Dashboard — launches your existing src/dashboard.py via
# `streamlit run`, the exact same way you already run it. The
# React page then just shows it in an iframe.
# ------------------------------------------------------------------
STREAMLIT_PROCESS = {"process": None}

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@app.route("/api/dashboard/start", methods=["POST"])
@token_required
def start_dashboard():
    if is_port_open("127.0.0.1", STREAMLIT_PORT):
        return jsonify({"status": "already_running", "url": f"http://localhost:{STREAMLIT_PORT}"})

    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            os.path.join("src", "dashboard.py"),
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ],
        cwd=PROJECT_ROOT,
    )
    STREAMLIT_PROCESS["process"] = process
    return jsonify({"status": "starting", "url": f"http://localhost:{STREAMLIT_PORT}"})


@app.route("/api/dashboard/status", methods=["GET"])
@token_required
def dashboard_status():
    running = is_port_open("127.0.0.1", STREAMLIT_PORT)
    return jsonify({"running": running, "url": f"http://localhost:{STREAMLIT_PORT}" if running else None})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
