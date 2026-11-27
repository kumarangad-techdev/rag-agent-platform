import { type FormEvent, useEffect, useRef, useState } from "react";
import { api, type ChatTurn, type Health, type Step } from "../api";

interface Message {
  role: "user" | "assistant";
  content: string;
  steps?: Step[];
  error?: boolean;
}

const LIVE_EXAMPLES = [
  "Which city has the most customers, and what share of all customers is that?",
  "What was the total revenue from Technology books, and what is 18% GST on it?",
  "Search the web for the latest LangGraph release and summarise it.",
];

function pretty(output: string): string {
  try {
    return JSON.stringify(JSON.parse(output), null, 2);
  } catch {
    return output;
  }
}

interface TableData {
  columns: string[];
  rows: unknown[][];
  truncated: boolean;
}

function asTable(output: string): TableData | null {
  try {
    const data = JSON.parse(output);
    return Array.isArray(data?.columns) && Array.isArray(data?.rows) ? data : null;
  } catch {
    return null;
  }
}

function ResultTable({ data }: { data: TableData }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {data.columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.truncated && <p className="muted small">Showing the first 50 rows.</p>}
    </div>
  );
}

function StepCard({ step, index }: { step: Step; index: number }) {
  const table = asTable(step.output);
  return (
    <details className={`step ${step.status}`} open={table !== null}>
      <summary>
        <span className="step-no">{index + 1}</span>
        <code>{step.tool}</code>
        <span className="muted small">{step.status === "error" ? "error" : "ok"}</span>
      </summary>
      <div className="step-body">
        <div className="label">Input</div>
        <pre>{JSON.stringify(step.args, null, 2)}</pre>
        <div className="label">Output</div>
        {table ? <ResultTable data={table} /> : <pre>{pretty(step.output)}</pre>}
      </div>
    </details>
  );
}

export default function AgentPanel({ health }: { health: Health | null }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const examples = health?.mode === "live" ? LIVE_EXAMPLES : (health?.offline_examples ?? []);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    const history: ChatTurn[] = messages.filter((m) => !m.error).map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: message }]);
    setInput("");
    setBusy(true);
    try {
      const reply = await api.agent(message, history);
      setMessages((m) => [...m, { role: "assistant", content: reply.answer, steps: reply.steps }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: (e as Error).message, error: true }]);
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void send(input);
  }

  return (
    <section className="card agent">
      <div className="agent-head">
        <div>
          <h2>Tool agent</h2>
          <p className="muted">
            LangGraph loop with four tools: <code>calculator</code>, <code>web_search</code>,{" "}
            <code>get_database_schema</code> and <code>query_database</code> (read-only SQLite bookshop).
          </p>
        </div>
        {messages.length > 0 && (
          <button className="button" onClick={() => setMessages([])}>
            Clear
          </button>
        )}
      </div>

      {health?.mode === "offline" && (
        <div className="alert info">
          Offline demo: a transparent rule-based router picks the tool, so only the example patterns below work.
          Configure an LLM for free-form questions.
        </div>
      )}

      <div className="chat" aria-live="polite">
        {messages.length === 0 && (
          <div className="examples">
            <p className="muted small">Try one of these:</p>
            {examples.map((e) => (
              <button key={e} className="chip" onClick={() => send(e)}>
                {e}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}${m.error ? " error" : ""}`}>
            {m.steps && m.steps.length > 0 && (
              <div className="steps">
                {m.steps.map((s, j) => (
                  <StepCard key={j} step={s} index={j} />
                ))}
              </div>
            )}
            <p>{m.content}</p>
          </div>
        ))}
        {busy && <div className="bubble assistant thinking">Working…</div>}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} className="ask">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the agent…"
          aria-label="Message"
        />
        <button className="button primary" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
