import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

export default function Reports() {
  const [files, setFiles] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    client.get("/api/reports")
      .then((res) => setFiles(res.data.files))
      .catch((err) => setErrorMsg(err.response?.data?.error || "Could not load reports"));
  }, []);

  async function download(name) {
    const token = localStorage.getItem("token");
    const res = await fetch(
      `${import.meta.env.VITE_API_URL}/api/reports/download/${encodeURIComponent(name)}`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    const blob = await res.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
  }

  return (
    <div>
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        📄 Reports
      </motion.h1>
      <p className="page-subtitle">Files generated into your reports/ folder</p>

      {errorMsg && <div className="auth-error">{errorMsg}</div>}

      <motion.div className="panel">
        {files.length === 0 && !errorMsg && <p>No reports yet — run the analysis first.</p>}
        {files.map((f, i) => (
          <motion.div
            key={f.name}
            className="report-row"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
          >
            <div>
              <strong>{f.name}</strong>
              <div className="report-meta">{f.size_kb} KB · {new Date(f.modified).toLocaleString()}</div>
            </div>
            <button className="btn-ghost" onClick={() => download(f.name)}>Download</button>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
