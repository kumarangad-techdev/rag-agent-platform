import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { type Answer, api, type DocumentInfo, type IngestResult } from "../api";

function describe(results: IngestResult[]): string {
  return results
    .map((r) =>
      r.error ? `${r.filename}: ${r.error}` : r.duplicate ? `${r.filename}: already indexed` : `${r.filename}: ${r.chunks} chunks`,
    )
    .join(" · ");
}

export default function DocumentsPanel({ onChange }: { onChange: () => void }) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState<{ kind: "info" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [asking, setAsking] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.documents());
      onChange();
    } catch (e) {
      setNotice({ kind: "error", text: (e as Error).message });
    }
  }, [onChange]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function ingest(action: () => Promise<IngestResult[]>) {
    setBusy(true);
    setNotice(null);
    try {
      const results = await action();
      setNotice({ kind: results.some((r) => r.error) ? "error" : "info", text: describe(results) });
      await refresh();
    } catch (e) {
      setNotice({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(doc: DocumentInfo) {
    try {
      await api.remove(doc.doc_id);
      selected.delete(doc.doc_id);
      setSelected(new Set(selected));
      await refresh();
    } catch (e) {
      setNotice({ kind: "error", text: (e as Error).message });
    }
  }

  function toggle(docId: string) {
    const next = new Set(selected);
    if (next.has(docId)) next.delete(docId);
    else next.add(docId);
    setSelected(next);
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    setAnswer(null);
    setNotice(null);
    try {
      setAnswer(await api.ask(question.trim(), topK, [...selected]));
    } catch (e) {
      setNotice({ kind: "error", text: (e as Error).message });
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="grid">
      <section className="card">
        <h2>Documents</h2>
        <p className="muted">PDF, TXT or Markdown. Files are chunked, embedded and stored in ChromaDB.</p>
        <div className="row">
          <label className="button primary">
            {busy ? "Indexing…" : "Upload files"}
            <input
              ref={fileInput}
              type="file"
              multiple
              accept=".pdf,.txt,.md,.markdown"
              hidden
              disabled={busy}
              aria-label="Upload files"
              onChange={(e) => e.target.files?.length && ingest(() => api.upload([...e.target.files!]))}
            />
          </label>
          <button className="button" disabled={busy} onClick={() => ingest(api.loadSamples)}>
            Load sample documents
          </button>
        </div>
        {notice && <div className={`alert ${notice.kind}`}>{notice.text}</div>}

        {docs.length === 0 ? (
          <p className="empty">No documents yet.</p>
        ) : (
          <>
            <p className="muted small">Tick documents to limit the search to them; none ticked searches all.</p>
            <ul className="doc-list">
              {docs.map((doc) => (
                <li key={doc.doc_id}>
                  <label>
                    <input type="checkbox" checked={selected.has(doc.doc_id)} onChange={() => toggle(doc.doc_id)} />
                    <span className={`type ${doc.file_type}`}>{doc.file_type}</span>
                    <span className="name">{doc.filename}</span>
                  </label>
                  <span className="muted small">{doc.chunks} chunks</span>
                  <button className="icon" aria-label={`Delete ${doc.filename}`} onClick={() => remove(doc)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="card">
        <h2>Ask your documents</h2>
        <form onSubmit={ask} className="ask">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. How many days of paid annual leave do employees get?"
            aria-label="Question"
          />
          <label className="topk" title="Number of chunks to retrieve">
            top-k
            <input type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          </label>
          <button className="button primary" disabled={asking || !question.trim()}>
            {asking ? "Searching…" : "Ask"}
          </button>
        </form>

        {answer && (
          <div className="answer">
            <p className="answer-text">{answer.answer}</p>
            {answer.warnings.map((w) => (
              <div key={w} className="alert warning">
                {w}
              </div>
            ))}
            {answer.sources.length > 0 && <h3>Retrieved sources</h3>}
            {answer.sources.map((s) => (
              <details key={s.marker} className="source">
                <summary>
                  <strong>[{s.marker}]</strong> {s.filename}
                  {s.page != null && ` · page ${s.page}`}
                  {s.heading && ` · ${s.heading}`}
                  <span className="score">{s.score.toFixed(3)}</span>
                </summary>
                <p>{s.text}</p>
              </details>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
