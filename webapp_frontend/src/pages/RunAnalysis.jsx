import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

export default function RunAnalysis() {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [logTail, setLogTail] = useState("");
  const [starting, setStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const pollRef = useRef(null);

  async function startRun() {
    setErrorMsg("");
    setStarting(true);
    try {
      const res = await client.post("/api/analysis/run");
      setJobId(res.data.job_id);
      setStatus("running");
    } catch (err) {
      setErrorMsg(err.response?.data?.error || "Could not start the analysis run");
    }
    setStarting(false);
  }

  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      const res = await client.get(`/api/analysis/status/${jobId}`);
      setStatus(res.data.status);
      setLogTail(res.data.log_tail);
      if (res.data.status !== "running") {
        clearInterval(pollRef.current);
      }
    }, 2000);
    return () => clearInterval(pollRef.current);
  }, [jobId]);

  return (
    <div>
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        ▶️ Run Analysis
      </motion.h1>
      <p className="page-subtitle">
        Runs your existing <code>main.py</code> pipeline: preprocessing → analysis → A/B testing → client report.
      </p>

      <motion.button
        className="btn-primary"
        whileTap={{ scale: 0.96 }}
        onClick={startRun}
        disabled={starting || status === "running"}
      >
        {status === "running" ? "Running..." : "Start Analysis"}
      </motion.button>

      {errorMsg && <div className="auth-error" style={{ marginTop: 16 }}>{errorMsg}</div>}

      {status && (
        <motion.div className="panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div className={"status-badge status-" + status}>{status.toUpperCase()}</div>
          <h3 style={{ marginTop: 16 }}>Live log (last 60 lines)</h3>
          <pre className="log-box">{logTail || "Waiting for output..."}</pre>
        </motion.div>
      )}
    </div>
  );
}
