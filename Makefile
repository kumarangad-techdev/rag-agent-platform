.PHONY: install ui dev run test lint smoke docker

install:            ## Python + UI dependencies
	python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[dev]"
	cd frontend && npm ci

ui:                 ## Build the React UI into frontend/dist (served by FastAPI)
	cd frontend && npm run build

run: ui             ## Build the UI and start the server on http://127.0.0.1:8000
	python -m rag_agent_platform

dev:                ## API on :8000 and Vite dev server on :5173 (hot reload)
	python -m rag_agent_platform & cd frontend && npm run dev

test:
	python -m pytest -q
	cd frontend && npm test

lint:
	python -m ruff check .
	python -m ruff format --check .

smoke:
	python scripts/smoke.py

docker:
	docker compose up --build
