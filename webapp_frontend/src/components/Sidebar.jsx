import React from "react";
import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { useAuth } from "../context/AuthContext.jsx";

const links = [
  { to: "/", label: "Home", icon: "🏠", end: true },
  { to: "/collect", label: "Collect Data", icon: "🎥" },
  { to: "/run-analysis", label: "Run Analysis", icon: "▶️" },
  { to: "/dataset", label: "Dataset", icon: "📊" },
  { to: "/dashboard", label: "Dashboard", icon: "🧠" },
  { to: "/recommendations", label: "AI Recommendations", icon: "✅" },
  { to: "/reports", label: "Reports", icon: "📄" },
];

export default function Sidebar() {
  const { username, logout } = useAuth();

  return (
    <motion.aside
      className="sidebar"
      initial={{ x: -40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="sidebar-brand">🧠 Neuro<span>Insights</span></div>
      <nav className="sidebar-nav">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) => "sidebar-link" + (isActive ? " active" : "")}
          >
            <span className="sidebar-icon">{link.icon}</span>
            {link.label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="sidebar-user">👤 {username}</div>
        <button className="btn-ghost" onClick={logout}>Log out</button>
      </div>
    </motion.aside>
  );
}
