import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import client from "../api/client";

export default function Dataset() {
  const [data, setData] = useState(null);
  const [page, setPage] = useState(1);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    let cancelled = false;
    client.get(`/api/dataset?page=${page}&page_size=25`)
      .then((res) => { if (!cancelled) setData(res.data); })
      .catch((err) => setErrorMsg(err.response?.data?.error || "Could not load dataset"));
    return () => { cancelled = true; };
  }, [page]);

  return (
    <div>
      <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        📊 Dataset
      </motion.h1>
      <p className="page-subtitle">Live view of data/unified_dataset_clean.csv</p>

      {errorMsg && <div className="auth-error">{errorMsg}</div>}

      {data && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <p>{data.total_rows.toLocaleString()} total rows</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{data.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i}>
                    {data.columns.map((c) => <td key={c}>{String(row[c])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <button className="btn-ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</button>
            <span>Page {page}</span>
            <button className="btn-ghost" disabled={page * 25 >= data.total_rows} onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </motion.div>
      )}
    </div>
  );
}
