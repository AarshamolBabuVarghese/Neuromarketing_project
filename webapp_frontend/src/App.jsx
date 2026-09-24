import React from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";
import Home from "./pages/Home.jsx";
import RunAnalysis from "./pages/RunAnalysis.jsx";
import CollectData from "./pages/CollectData.jsx";
import Dataset from "./pages/Dataset.jsx";
import StreamlitDashboard from "./pages/StreamlitDashboard.jsx";
import Recommendations from "./pages/Recommendations.jsx";
import Reports from "./pages/Reports.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";

function AppLayout({ children }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">{children}</main>
    </div>
  );
}

function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25 }}
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/login" element={<PageTransition><Login /></PageTransition>} />
        <Route path="/register" element={<PageTransition><Register /></PageTransition>} />

        <Route path="/" element={
          <ProtectedRoute><AppLayout><PageTransition><Home /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/collect" element={
          <ProtectedRoute><AppLayout><PageTransition><CollectData /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/run-analysis" element={
          <ProtectedRoute><AppLayout><PageTransition><RunAnalysis /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/dataset" element={
          <ProtectedRoute><AppLayout><PageTransition><Dataset /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute><AppLayout><PageTransition><StreamlitDashboard /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/recommendations" element={
          <ProtectedRoute><AppLayout><PageTransition><Recommendations /></PageTransition></AppLayout></ProtectedRoute>
        } />
        <Route path="/reports" element={
          <ProtectedRoute><AppLayout><PageTransition><Reports /></PageTransition></AppLayout></ProtectedRoute>
        } />
      </Routes>
    </AnimatePresence>
  );
}
