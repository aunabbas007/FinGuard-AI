# FinGuard AI

**An AI-assisted fraud detection and investigation platform built with synthetic financial data.**

FinGuard AI combines a machine-learning fraud scoring pipeline, temporal account-network features, a FastAPI service, a browser dashboard, and a LangChain investigation assistant. It is a portfolio and learning project—not a banking product.

**Live demo:** https://finguard-ai-x00k.onrender.com  
**Repository:** https://github.com/aunabbas007/FinGuard-AI  
**Deployment branch:** `feature/full-stack-app`

> **Safety notice:** FinGuard AI uses synthetic data only. It does not connect to banks, initiate or block payments, freeze accounts, or make real financial decisions. Risk scores are not calibrated real-world probabilities. All flags are triage suggestions that require human review.

---

## What it does

- Generates a reproducible synthetic transaction dataset for model development.
- Builds transaction-level and temporal graph features from account activity.
- Trains and evaluates Random Forest fraud classifiers using chronological train/validation/test splits.
- Exposes model scoring and transaction-assessment history through a FastAPI backend.
- Provides a browser dashboard for synthetic demo scoring and reviewing saved assessments.
- Includes a LangChain investigation agent powered by Groq, with tools for retrieving fraud guidance and looking up recent or referenced assessments.
- Runs in Docker and includes a GitHub Actions workflow for automated tests and Docker image builds.

## Architecture

```text
Synthetic transaction data
          |
          v
Feature engineering ------> Temporal graph features
          |                           |
          +-------------+-------------+
                        |
                        v
             Random Forest models
                        |
                        v
                 FastAPI backend
                 /             \
                v               v
        Browser dashboard   SQLite / PostgreSQL
                |
                v
       Investigation assistant
          |             |
          v             v
     LangChain + Groq   TF-IDF retrieval
          |
          v
   Transaction lookup tools
```

### Technology stack

| Area | Technologies |
|---|---|
| Language and data | Python 3.11, pandas, NumPy |
| Machine learning | scikit-learn, Random Forest |
| Graph analytics | NetworkX |
| Explainability exploration | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Persistence | SQLAlchemy, SQLite by default, optional PostgreSQL |
| LLM assistant | LangChain, LangChain-Groq, Groq |
| Retrieval | TF-IDF and cosine similarity over a small curated fraud-guidance corpus |
| Frontend | HTML, CSS, JavaScript |
| Packaging and deployment | Docker, Docker Compose, Render |
| CI | GitHub Actions, pytest |

## Fraud model and evaluation

The project includes a baseline feature set and a graph-enhanced feature set. The graph features summarize prior account activity in time order; they are intended to add relationship and behavioral context to transaction scoring.

Model evaluation uses chronological train, validation, and test partitions to reduce temporal leakage. The project evaluates ranking metrics and threshold-based classification metrics. Because the dataset is synthetic and the held-out fraud sample is small, measured performance is illustrative and should not be interpreted as expected real-world fraud detection performance.

The deployed Docker image regenerates the synthetic data and trains the baseline model during image build. Generated model binaries and datasets do not need to be committed to the repository.

## Investigation assistant: what is and is not implemented

The investigation assistant uses LangChain's `create_agent` with a Groq chat model and controlled tools to:

- Retrieve relevant fraud-pattern and investigation guidance.
- Look up the latest saved transaction assessments.
- Look up an assessment by its numeric assessment ID or external transaction reference.

The current retrieval implementation is **TF-IDF + cosine similarity** over a small curated knowledge base. It is a lightweight retrieval-augmented generation (RAG) prototype; it is **not** a ChromaDB/Pinecone vector database implementation. The agent uses LangChain's agent runtime; this repository does not define a separate custom LangGraph `StateGraph` workflow.

The assistant supports analyst investigation. It must not be treated as an autonomous decision-maker.

## Run locally (Windows PowerShell)

Use Python 3.11.

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

Open:

- Dashboard: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The data and model-generation commands create local artifacts required by the application. They are generated locally and are not committed to Git.

## Run with Docker

```powershell
docker compose up --build
```

Open http://localhost:8000. Docker Compose uses a named volume for the SQLite database. The Docker image builds the synthetic dataset and baseline model as part of its build process.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service and database health |
| GET | `/api/v1` | API overview |
| GET | `/api/v1/model/schema` | Model feature schema and scoring threshold |
| GET | `/api/v1/demo/sample` | Random synthetic transaction with its aligned feature vector |
| POST | `/api/v1/transactions/score` | Score and save a transaction assessment |
| GET | `/api/v1/transactions` | Paginated assessment history |
| GET | `/api/v1/transactions?flagged_only=true` | Assessments flagged for review |
| GET | `/api/v1/transactions/{transaction_id}` | Retrieve an assessment by database ID |
| POST | `/api/v1/investigation/chat` | Ask the investigation assistant a question |

The scoring endpoint expects the complete numeric feature vector defined by `/api/v1/model/schema`. The dashboard's demo action obtains a synthetic transaction and its matching feature vector from the demo endpoint before scoring it.

## Configuration

Set the following environment variables in a local `.env` file or in the Render service environment:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | For LLM chat | Secret API key for Groq. Never commit it. |
| `GROQ_MODEL` | No | Groq model identifier. Defaults to `openai/gpt-oss-120b`. |
| `DATABASE_URL` | No | SQLAlchemy database URL. Defaults to SQLite. |

Without `GROQ_API_KEY`, the dashboard and fraud-scoring API remain available, but live LLM investigation responses are unavailable.

## Tests and CI

Run tests locally:

```powershell
pytest -q
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on pull requests targeting `main`, pushes to `main`, `feature/**`, and `fix/**`, and manual workflow dispatch. It installs Python 3.11 dependencies, runs pytest, and builds the Docker image.

## Deployment

The live Render web service is configured to deploy from **`feature/full-stack-app`**. The repository also contains a `render.yaml` Blueprint configuration.

Live service: https://finguard-ai-x00k.onrender.com  
Render dashboard: https://dashboard.render.com/web/srv-dbu26ckipgvc73agepi0

The current demo configuration uses SQLite on Render's filesystem. Unless a persistent disk or managed database is configured, stored transaction history may be lost when the service restarts or redeploys. A managed PostgreSQL database or persistent disk may incur charges; review pricing before enabling paid resources.

## Limitations and responsible use

- The dataset is synthetic; no real customer or banking data is included.
- Model scores are not calibrated probabilities and do not establish that a transaction is fraudulent.
- Threshold-based flags are suggestions for human investigation only.
- SHAP and feature-importance outputs are exploratory model explanations, not proof of causation.
- The knowledge base is small and curated; retrieval is not a production policy or compliance system.
- The project does not implement real bank integrations, payment controls, authentication, role-based access control, audit-grade logging, production-grade monitoring, or a formal security review.
- Do not submit real account numbers, card details, passwords, OTPs, or other sensitive financial information.

## Project status

**Portfolio release — feature-frozen.** The current deployed feature set is being preserved as a completed demonstration project. Further feature development is intentionally out of scope for this release; only critical fixes or security corrections should be considered.

---

Built as an AI/ML engineering portfolio project.
