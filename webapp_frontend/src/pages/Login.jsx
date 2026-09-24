import React, { useState } from "react";
import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Login() {
  const { login, error } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    const ok = await login(username, password);
    setLoading(false);
    if (ok) navigate("/");
  }

  return (
    <div className="auth-page">
      <motion.form
        className="auth-card"
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 30, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <h1>🧠 Welcome back</h1>
        <p className="auth-subtitle">Sign in to your Neuromarketing workspace</p>

        <label>Username</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />

        <label>Password</label>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />

        {error && <div className="auth-error">{error}</div>}

        <motion.button
          className="btn-primary"
          type="submit"
          whileTap={{ scale: 0.96 }}
          disabled={loading}
        >
          {loading ? "Signing in..." : "Sign in"}
        </motion.button>

        <p className="auth-switch">
          No account yet? <Link to="/register">Register</Link>
        </p>
      </motion.form>
    </div>
  );
}
