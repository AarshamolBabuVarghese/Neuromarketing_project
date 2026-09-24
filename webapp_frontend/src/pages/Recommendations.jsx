import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

const BADGE_CLASS = { High: "badge-high", Medium: "badge-medium", Low: "badge-low" };

export default function Recommendations() {
  const [recs, setRecs] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    client.get("/api/recommendations")
      .then((res) => setRecs(res.data.recommendations))
      .catch((err) => setErrorMsg(err.response?.data?.error || "Could not load recommendations"));
  }, []);

  return (
    <div>
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        ✅ AI Recommendations
      </motion.h1>
      <p className="page-subtitle">Per-product, per-attribute A/B winners from rec.py</p>

      {errorMsg && <div className="auth-error">{errorMsg}</div>}

      <motion.div className="card-grid" initial="hidden" animate="show">
        {recs.map((r, i) => (
          <motion.div
            key={i}
            className="rec-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.03 }}
          >
            <div className={"badge " + (BADGE_CLASS[r.confidence] || "")}>{r.confidence}</div>
            <h3>{r.product_name}</h3>
            <p className="rec-attr">{String(r.attribute_changed).replace(/_/g, " ")}</p>
            <p>{r.winner_variant === "No clear winner"
              ? "No significant preference found yet."
              : <>Adopt <strong>{r.winner_value}</strong></>}
            </p>
            {r.recommendation && <p className="rec-note">{r.recommendation}</p>}
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
