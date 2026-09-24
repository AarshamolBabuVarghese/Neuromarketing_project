import React, { createContext, useContext, useState } from "react";
import client from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [username, setUsername] = useState(localStorage.getItem("username") || null);
  const [error, setError] = useState("");

  async function login(username, password) {
    setError("");
    try {
      const res = await client.post("/api/login", { username, password });
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("username", res.data.username);
      setUsername(res.data.username);
      return true;
    } catch (err) {
      setError(err.response?.data?.error || "Login failed");
      return false;
    }
  }

  async function register(username, password) {
    setError("");
    try {
      const res = await client.post("/api/register", { username, password });
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("username", res.data.username);
      setUsername(res.data.username);
      return true;
    } catch (err) {
      setError(err.response?.data?.error || "Registration failed");
      return false;
    }
  }

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    setUsername(null);
  }

  return (
    <AuthContext.Provider value={{ username, error, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
