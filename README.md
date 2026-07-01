# RAG + LLM-as-Judge Repository

This repository implements:

1. A cost-efficient RAG service using LanceDB.
2. An LLM-as-judge evaluation pipeline with bias mitigation.

## Quickstart

1. Copy `.env.example` to `.env` and fill in values.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the API: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

## Structure

- `app/` core FastAPI app and modules
- `evaluation/` retrieval and judge evaluation assets
- `tests/` unit tests
- `docs/` reports and docs
