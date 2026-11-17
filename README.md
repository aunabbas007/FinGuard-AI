# FinGuard AI

**Synthetic-data fraud detection and investigation demo with a FastAPI backend and browser dashboard.**

FinGuard AI is an educational portfolio project. It does not connect to banks, initiate payments, freeze accounts, or make real financial decisions. Model scores are not calibrated real-world probabilities and must not be used to approve, reject, or block financial activity.

## Features
- Reproducible synthetic dataset generation (10,000 transactions).
- Baseline and graph-enhanced fraud-model evaluation with chronological train/validation/test splits.
- FastAPI service with health, model-schema, synthetic-demo, transaction scoring, and transaction-history endpoints.
- SQLite persistence by default; PostgreSQL can be configured through DATABASE_URL.
- Browser dashboard with one-click synthetic sample loading and automatic fraud assessment, risk results, and recent transaction history.
- LangChain investigation agent with Groq chat model, tool-based transaction lookup, and retrieval-augmented fraud guidance (TF-IDF over the curated local knowledge base).
- Docker image builds the synthetic dataset and baseline model itself; no model binary needs to be committed.
- Docker Compose and Render Blueprint deployment configuration.

## Run locally (Windows PowerShell)

Use Python 3.11:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/data/generate_transactions_v2.py
python src/features/build_features.py
python src/graph/build_graph_features.py
python src/features/merge_graph_features.py
python src/models/evaluate_models.py
uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000 for the dashboard, http://127.0.0.1:8000/docs for API docs, and http://127.0.0.1:8000/health for service health.

The ML commands generate models/fraud_detection_model.joblib and models/thresholds.json. These are local generated artifacts and are not committed to Git. The Docker build runs the deterministic synthetic-data and baseline-model pipeline automatically.

## Docker
```powershell
docker compose up --build
```
Open http://localhost:8000. The named Docker volume stores the SQLite database. The image trains its synthetic baseline model during build.

## API
- GET /health — API and database health.
- GET /api/v1/model/schema — exact feature names and validation threshold.
- GET /api/v1/demo/sample — one randomly selected synthetic transaction and its aligned model feature vector.
- POST /api/v1/transactions/score — score and persist a transaction using a complete feature vector.
- GET /api/v1/transactions — paginated assessment history.
- GET /api/v1/transactions?flagged_only=true — flagged assessments only.
- GET /api/v1/transactions/{id} — retrieve one saved assessment.
- POST /api/v1/investigation/chat — ask the LangChain investigation agent a question; include a numeric transaction database ID in the message to let its lookup tool inspect a saved assessment.

The scoring API deliberately requires the model's exact numeric feature schema. Raw transaction metadata alone is not enough to recreate historical velocity and account-network features. The dashboard's demo button loads a known-aligned synthetic feature row and automatically submits it to the scoring endpoint. The investigation assistant retrieves relevant fraud guidance and can look up saved transactions by database ID. It is an analyst-support feature, not an autonomous fraud decision system.

## Tests
```powershell
pytest -q
```
GitHub Actions runs the tests and builds the Docker image on pull requests and feature-branch pushes.

## AI investigation assistant configuration

The agent uses LangChain with Groq. Add these environment variables to the Render service (or local `.env`) to enable live LLM responses:

- `GROQ_API_KEY` — API key from your Groq account. Keep it secret; never commit it to Git.
- `GROQ_MODEL` — optional; defaults to `llama-3.3-70b-versatile`.

Without `GROQ_API_KEY`, the dashboard and fraud-scoring API continue to work, while the investigation endpoint returns HTTP 503 with a configuration message. Retrieval uses TF-IDF over the curated fraud guidance in `data/knowledge` and is performed before the LLM synthesizes an answer. The agent has tools for knowledge retrieval and lookup of saved transaction records.

## Deployment (Render)
The repository includes render.yaml. After merging the application branch into main, connect the GitHub repository to Render and create a Blueprint deployment from that file. Render builds the Docker image and runs the service health check at /health.

The included free Render configuration uses SQLite on an ephemeral filesystem. It is suitable for a portfolio demo, but transaction history may be lost when the service restarts or redeploys. For durable history, attach a persistent disk or configure a managed PostgreSQL database; those options may incur charges. Do not enable paid resources without reviewing their price first.

## Important limitations
- Synthetic training data only; no real banking dataset or bank integration.
- Model output is a ranking score, not a calibrated probability.
- Threshold-based flags are investigation suggestions only; human review is required.
- The demo transaction button samples synthetic records and is not a raw-transaction feature-engineering service.
- The investigation agent depends on a configured Groq API key; without it, LLM responses are unavailable.
- Retrieval is a small curated knowledge-base prototype, not a production fraud-policy corpus or a vector database.
- Authentication, role-based access control, audit-grade event logging, production monitoring, migrations, and security review are not implemented. Do not expose sensitive or real customer data.