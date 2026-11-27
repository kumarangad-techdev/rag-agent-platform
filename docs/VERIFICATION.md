# Verification

What was actually checked for this revision, and what was not.

## Checked (Linux, Python 3.12, Node 22)

| Check | Result |
|---|---|
| `python -m pytest -q` | 80 tests pass |
| `python -m ruff check .` and `ruff format --check .` | clean |
| `python scripts/smoke.py` | all checks pass |
| `npm test` (Vitest + Testing Library, jsdom) | 4 tests pass |
| `npm run build` | builds |
| Real server + built UI in headless Chrome | samples load, Q&A shows cited sources, agent runs SQL and calculator steps |

The tests cover chunk size and overlap, Markdown and PDF loading, rejected file types, duplicate
uploads, persistence across restarts, refusing a different embedder on existing data, retrieval
filtered by document, offline and live answers, unknown or missing citations, provider failures,
calculator injection attempts, SQL write and `ATTACH`/`PRAGMA`/`load_extension` attempts,
runaway queries, row limits, the offline router, multi-step live tool chains, the agent step
limit, upload size limits and SPA static serving.

## Not checked here

* **Hugging Face model download.** The build environment had no access to huggingface.co, so the
  tests and smoke run used the hashing embedder. `HuggingFaceEmbedder` was tested against a fake
  `sentence_transformers` module. Run the app once with internet access to download the model.
* **Real LLM calls.** Live mode was tested with a scripted fake chat model that returns tool calls
  and answers. No OpenAI or Anthropic requests were made with real credentials.
* **Real web search.** DuckDuckGo was not reachable from the build environment; the tool was tested
  with a fake client.
* **Docker.** No Docker daemon was available. The Python quickstart is the verified path.
* **Windows and macOS** were not tested.
