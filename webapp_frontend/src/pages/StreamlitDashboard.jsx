import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

export default function StreamlitDashboard() {
  const [url, setUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        await client.post("/api/dashboard/start");
        // give Streamlit a moment to boot on first launch
        for (let i = 0; i < 15; i++) {
          const res = await client.get("/api/dashboard/status");
          if (res.data.running) {
            if (!cancelled) { setUrl(res.data.url); setLoading(false); }
            return;
          }
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (!cancelled) setErrorMsg("Streamlit is taking longer than expected to start.");
      } catch (err) {
        if (!cancelled) setErrorMsg(err.response?.data?.error || "Could not start the dashboard");
      }
      if (!cancelled) setLoading(false);
    }
    boot();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="dashboard-page">
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        🧠 Dashboard
      </motion.h1>
      <p className="page-subtitle">Your existing Streamlit dashboard, launched automatically.</p>

      {loading && <div className="panel">Starting Streamlit dashboard...</div>}
      {errorMsg && <div className="auth-error">{errorMsg}</div>}

      {url && (
        <motion.iframe
          src={url}
          title="Streamlit Dashboard"
          className="streamlit-frame"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        />
      )}
    </div>
  );
}
