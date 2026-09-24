import React, { useState } from "react";
import { motion } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function Register() {
  const { register, error } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    const ok = await register(username, password);
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
        <h1>✨ Create your account</h1>
        <p className="auth-subtitle">Set up access to your Neuromarketing workspace</p>

        <label>Username</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />

        <label>Password</label>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6} />
        <p className="auth-hint">At least 6 characters</p>

        {error && <div className="auth-error">{error}</div>}

        <motion.button
          className="btn-primary"
          type="submit"
          whileTap={{ scale: 0.96 }}
          disabled={loading}
        >
          {loading ? "Creating account..." : "Create account"}
        </motion.button>

        <p className="auth-switch">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </motion.form>
    </div>
  );
}
