import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const health = {
  status: "ok",
  mode: "offline",
  provider: "offline",
  model: null,
  embeddings: "huggingface:sentence-transformers/all-MiniLM-L6-v2",
  documents: 1,
  chunks: 3,
  web_search: true,
  offline_examples: ["calculate 2 + 2", "revenue by category"],
};

const docs = [
  { doc_id: "d1", filename: "handbook.md", file_type: "md", uploaded_at: "2026-01-01T00:00:00", chunks: 3 },
];

function mockFetch(routes: Record<string, unknown>) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    if (!(key in routes)) return new Response(JSON.stringify({ detail: `no route ${key}` }), { status: 404 });
    const body = routes[key];
    return new Response(JSON.stringify(typeof body === "function" ? body(init) : body), { status: 200 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App", () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => vi.unstubAllGlobals());

  it("shows the mode badge and indexed documents", async () => {
    mockFetch({ "GET /api/health": health, "GET /api/documents": docs });
    render(<App />);
    expect(await screen.findByText("Offline demo · no LLM")).toBeInTheDocument();
    expect(await screen.findByText("handbook.md")).toBeInTheDocument();
    expect(screen.getByText("3 chunks")).toBeInTheDocument();
  });

  it("asks a question and shows cited sources", async () => {
    const fetchMock = mockFetch({
      "GET /api/health": health,
      "GET /api/documents": docs,
      "POST /api/qa": {
        answer: "Employees get 24 days [S1].",
        mode: "live",
        warnings: [],
        sources: [
          {
            marker: "S1",
            doc_id: "d1",
            filename: "handbook.md",
            chunk_index: 2,
            score: 0.8123,
            text: "Every employee receives 24 days of paid annual leave.",
            page: null,
            heading: "Leave policy",
          },
        ],
      },
    });
    render(<App />);
    await screen.findByText("handbook.md");
    await userEvent.type(screen.getByLabelText("Question"), "How much leave?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Employees get 24 days [S1].")).toBeInTheDocument();
    expect(screen.getByText("0.812")).toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([url]) => url === "/api/qa")!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({ question: "How much leave?", top_k: 4, doc_ids: null });
  });

  it("runs the agent and shows tool steps", async () => {
    mockFetch({
      "GET /api/health": health,
      "GET /api/documents": docs,
      "POST /api/agent": {
        answer: "Result: 4",
        mode: "offline",
        steps: [{ tool: "calculator", args: { expression: "2 + 2" }, output: "4", status: "success" }],
      },
    });
    render(<App />);
    await userEvent.click(screen.getByRole("tab", { name: "Tool Agent" }));
    await userEvent.click(await screen.findByRole("button", { name: "calculate 2 + 2" }));

    expect(await screen.findByText("Result: 4")).toBeInTheDocument();
    expect(screen.getByText("calculator", { selector: "summary code" })).toBeInTheDocument();
  });

  it("shows backend errors instead of failing silently", async () => {
    mockFetch({ "GET /api/documents": docs });
    render(<App />);
    expect(await screen.findByText(/Backend not reachable/)).toBeInTheDocument();
  });
});
