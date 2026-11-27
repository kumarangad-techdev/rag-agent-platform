import { useCallback, useEffect, useState } from "react";
import { api, type Health } from "./api";
import AgentPanel from "./components/AgentPanel";
import DocumentsPanel from "./components/DocumentsPanel";

type Tab = "docs" | "agent";

export default function App() {
  const [tab, setTab] = useState<Tab>("docs");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshHealth = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setError(null);
      })
      .catch((e: Error) => setError(`Backend not reachable: ${e.message}`));
  }, []);

  useEffect(refreshHealth, [refreshHealth]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>RAG Agent Platform</h1>
          <p className="subtitle">Document Q&amp;A with citations · LangGraph tool agent</p>
        </div>
        {health && (
          <div className="badges">
            <span className={`badge ${health.mode}`} title="Whether a language model is configured">
              {health.mode === "live" ? `Live · ${health.provider} · ${health.model}` : "Offline demo · no LLM"}
            </span>
            <span className="badge neutral" title="Embedding model">
              {health.embeddings}
            </span>
          </div>
        )}
      </header>

      {error && <div className="alert error">{error}</div>}

      <nav className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "docs"} onClick={() => setTab("docs")}>
          Document Q&amp;A
        </button>
        <button role="tab" aria-selected={tab === "agent"} onClick={() => setTab("agent")}>
          Tool Agent
        </button>
      </nav>

      <main>
        {tab === "docs" ? (
          <DocumentsPanel onChange={refreshHealth} />
        ) : (
          <AgentPanel health={health} />
        )}
      </main>
    </div>
  );
}
