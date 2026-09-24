import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

export default function ProtectedRoute({ children }) {
  const { username } = useAuth();
  if (!username) {
    return <Navigate to="/login" replace />;
  }
  return children;
}
