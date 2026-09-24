import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const options = [
  { to: "/collect", title: "Collect Data", desc: "Run a live A/B session: calibration, category picker, and webcam capture — right in the browser.", icon: "🎥" },
  { to: "/run-analysis", title: "Run Analysis", desc: "Kick off the full pipeline: preprocessing, analysis, A/B testing, and the client report.", icon: "▶️" },
  { to: "/dataset", title: "Dataset", desc: "Browse the unified, cleaned dataset behind every chart and recommendation.", icon: "📊" },
  { to: "/dashboard", title: "Dashboard", desc: "Open your full Streamlit BI dashboard, right here.", icon: "🧠" },
  { to: "/recommendations", title: "AI Recommendations", desc: "See which product variants are winning, by category and confidence.", icon: "✅" },
  { to: "/reports", title: "Reports", desc: "Download generated client reports and summaries.", icon: "📄" },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};
const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function Home() {
  const { username } = useAuth();
  const navigate = useNavigate();

  return (
    <div>
      <motion.h1
        className="page-title"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        Welcome back, {username} 👋
      </motion.h1>
      <p className="page-subtitle">What would you like to do today?</p>

      <motion.div className="card-grid" variants={container} initial="hidden" animate="show">
        {options.map((opt) => (
          <motion.div
            key={opt.to}
            className="option-card"
            variants={item}
            whileHover={{ y: -6, boxShadow: "0 16px 30px rgba(0,0,0,0.18)" }}
            onClick={() => navigate(opt.to)}
          >
            <div className="option-icon">{opt.icon}</div>
            <h3>{opt.title}</h3>
            <p>{opt.desc}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
