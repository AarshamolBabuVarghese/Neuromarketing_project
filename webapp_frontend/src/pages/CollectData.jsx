import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

export default function CollectData() {
  const [step, setStep] = useState("setup"); // setup | calibration | category | session | done
  const [userId, setUserId] = useState("");
  const [config, setConfig] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [queue, setQueue] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [alert, setAlert] = useState(null);
  const [calibPointIndex, setCalibPointIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [lastSavedRow, setLastSavedRow] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const captureTimerRef = useRef(null);
  const queueRef = useRef([]);
  const indexRef = useRef(0);

  useEffect(() => {
    client.get("/api/live/config").then((res) => setConfig(res.data));
    return () => { stopWebcam(); clearInterval(captureTimerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 800, height: 500 } });
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
  }

  function stopWebcam() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }

  function grabFrame(quality = 0.7) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  }

  // ---------- Step 1: setup ----------
  async function beginSetup() {
    setErrorMsg("");
    if (!userId.trim()) { setErrorMsg("Enter a participant ID first"); return; }
    try {
      await startWebcam();
      const res = await client.get("/api/live/manifest");
      setManifest(res.data);
      setStep("calibration");
      await new Promise((r) => setTimeout(r, 400)); // let the video element paint a frame
      startCalibrationPoint(0);
    } catch (err) {
      setErrorMsg("Could not access your webcam. Check browser camera permissions for this site.");
    }
  }

  // ---------- Step 2: calibration ----------
  async function startCalibrationPoint(index) {
    if (index === 0) {
      await client.post("/api/live/calibration/start", { user_id: userId });
    }
    setCalibPointIndex(index);

    const sampleMs = (config?.calibration_sample_seconds || 1.5) * 1000;
    const start = Date.now();

    // Self-paced loop: waits for each request to finish before sending
    // the next one, so it never queues up faster than the backend
    // (running your real DeepFace/MediaPipe code) can keep up with.
    while (Date.now() - start < sampleMs) {
      const frame = grabFrame();
      if (frame) {
        try {
          await client.post("/api/live/calibration/sample", { user_id: userId, point_index: index, frame });
        } catch (e) { /* skip a dropped frame */ }
      }
      await new Promise((r) => setTimeout(r, 30));
    }

    const totalPoints = config?.calibration_points?.length || 5;
    if (index + 1 < totalPoints) {
      startCalibrationPoint(index + 1);
    } else {
      finishCalibration();
    }
  }

  async function finishCalibration() {
    try {
      await client.post("/api/live/calibration/finish", { user_id: userId });
    } catch (err) {
      setErrorMsg(err.response?.data?.error || "Calibration had too few points — continuing uncalibrated.");
    }
    setStep("category");
  }

  // ---------- Step 3: category picker ----------
  function chooseCategory(cat) {
    const rows = cat === "__all__" ? manifest.rows : manifest.rows.filter((r) => r.category === cat);
    queueRef.current = rows;
    indexRef.current = 0;
    setQueue(rows);
    setCurrentIndex(0);
    setStep("session");
    runTrial(rows[0]);
  }

  // ---------- Step 4: trial loop ----------
  async function runTrial(row) {
    setAlert(null);
    const label = `${row.product_name}_${row.attribute_changed}`;
    const res = await client.post("/api/live/trial/start", { user_id: userId, product_feature_label: label });
    const trialMs = (res.data.trial_seconds || config.trial_seconds) * 1000;

    const start = Date.now();
    // Same self-paced pattern as calibration: wait for each frame's
    // result before sending the next, so it paces itself to real
    // processing speed instead of queueing up a backlog. The loop
    // still exits at the real trial duration either way.
    while (Date.now() - start < trialMs) {
      const frame = grabFrame();
      if (frame) {
        try {
          const fres = await client.post("/api/live/trial/frame", { frame });
          setAlert(fres.data.alert);
        } catch (e) { /* skip a dropped frame */ }
      }
      await new Promise((r) => setTimeout(r, 30));
    }
    finishTrial();
  }

  async function finishTrial() {
    try {
      const res = await client.post("/api/live/trial/finish");
      setLastSavedRow(res.data.saved_row);
    } catch (err) {
      setErrorMsg(err.response?.data?.error || "This trial couldn't be saved — moving to the next one.");
    }
    const nextIndex = indexRef.current + 1;
    if (nextIndex < queueRef.current.length) {
      indexRef.current = nextIndex;
      setCurrentIndex(nextIndex);
      setTimeout(() => runTrial(queueRef.current[nextIndex]), 800);
    } else {
      await client.post("/api/live/session/log", {
        user_id: userId,
        note: `Browser A/B session, ${queueRef.current.length} stimuli`,
      });
      stopWebcam();
      setStep("done");
    }
  }

  const currentRow = queue[currentIndex];

  return (
    <div>
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        🎥 Collect Data
      </motion.h1>
      <p className="page-subtitle">Your existing calibration, gaze and emotion logic — running from the browser.</p>

      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {errorMsg && <div className="auth-error">{errorMsg}</div>}

      {step === "setup" && (
        <div className="panel">
          <label>Participant ID</label>
          <input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="e.g. 101" />
          <motion.button className="btn-primary" whileTap={{ scale: 0.96 }} onClick={beginSetup}>
            Start Session
          </motion.button>
        </div>
      )}

      {step === "calibration" && config && (
        <div className="calib-stage">
          <p className="calib-label">
            Calibrating — look at the red dot ({calibPointIndex + 1}/{config.calibration_points.length})
          </p>
          <div
            className="calib-dot"
            style={{
              left: `${config.calibration_points[calibPointIndex][0] * 100}%`,
              top: `${config.calibration_points[calibPointIndex][1] * 100}%`,
            }}
          />
        </div>
      )}

      {step === "category" && manifest && (
        <motion.div className="card-grid" initial="hidden" animate="show">
          {manifest.categories.map((cat) => (
            <motion.div key={cat} className="option-card" whileHover={{ y: -4 }} onClick={() => chooseCategory(cat)}>
              <h3>{cat}</h3>
              <p>Test only this category</p>
            </motion.div>
          ))}
          <motion.div className="option-card" whileHover={{ y: -4 }} onClick={() => chooseCategory("__all__")}>
            <h3>All categories</h3>
            <p>{manifest.rows.length} stimuli total</p>
          </motion.div>
        </motion.div>
      )}

      {step === "session" && currentRow && (
        <div className="session-stage">
          {alert && <motion.div className="alert-banner" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>{alert}</motion.div>}
          <img
            className="stimulus-image"
            src={`${import.meta.env.VITE_API_URL}/api/live/stimulus-image/${currentRow.image_filename}`}
            alt="stimulus"
          />
          <p className="session-progress">Stimulus {currentIndex + 1} of {queue.length} — {currentRow.test_name}</p>
        </div>
      )}

      {step === "done" && (
        <motion.div className="panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h3>✅ Session complete</h3>
          <p>{queue.length} stimuli recorded for participant {userId}.</p>
          {lastSavedRow && <pre className="log-box">{JSON.stringify(lastSavedRow, null, 2)}</pre>}
        </motion.div>
      )}
    </div>
  );
}
