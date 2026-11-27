// Typed client for the FastAPI backend. All paths are relative, so the same build works when
// served by FastAPI and behind the Vite dev-server proxy.

export type Mode = "offline" | "live";

export interface Health {
  mode: Mode;
  provider: string;
  model: string | null;
  embeddings: string;
  documents: number;
  chunks: number;
  web_search: boolean;
  offline_examples: string[];
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  uploaded_at: string;
  chunks: number;
}

export interface IngestResult {
  doc_id: string | null;
  filename: string;
  chunks: number;
  duplicate: boolean;
  error: string | null;
}

export interface Source {
  marker: string;
  doc_id: string;
  filename: string;
  chunk_index: number;
  score: number;
  text: string;
  page: number | null;
  heading: string | null;
}

export interface Answer {
  answer: string;
  mode: Mode;
  sources: Source[];
  warnings: string[];
}

export interface Step {
  tool: string;
  args: Record<string, unknown>;
  output: string;
  status: string;
}

export interface AgentReply {
  answer: string;
  mode: Mode;
  steps: Step[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  documents: () => request<DocumentInfo[]>("/api/documents"),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<IngestResult[]>("/api/documents", { method: "POST", body: form });
  },
  loadSamples: () => request<IngestResult[]>("/api/documents/samples", { method: "POST" }),
  remove: (docId: string) => request<void>(`/api/documents/${docId}`, { method: "DELETE" }),
  ask: (question: string, topK: number, docIds?: string[]) =>
    request<Answer>("/api/qa", json({ question, top_k: topK, doc_ids: docIds?.length ? docIds : null })),
  agent: (message: string, history: ChatTurn[]) => request<AgentReply>("/api/agent", json({ message, history })),
};
