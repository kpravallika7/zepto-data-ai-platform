# Zepto Data & AI Platform — Capstone Project

An end-to-end AI/ML platform built for Zepto's analytics guild, made of three connected modules that live in this single repository:

| Module | Folder | Marks | What it does |
|---|---|---|---|
| 1. Data Pipeline | [`/data_pipeline`](./data_pipeline) | 25 | Scrapes book catalog data, cleans and converts it, stores it in a relational SQLite database, and queries it with SQL and pandas. |
| 2. Analytics Pipeline | [`/analytics`](./analytics) | 50 | Profiles and cleans the Titanic dataset, tells a visual data story, then builds, tunes, and evaluates a full classification + regression modeling pipeline. |
| 3. Support Assistant | [`/support_assistant`](./support_assistant) | 25 | A RAG-based GenAI assistant that answers Zepto policy questions, grounded in Zepto's own documents, served through a LangGraph workflow and a FastAPI endpoint. |

The three modules are graded independently but are meant to read as one story: a data pipeline feeds clean structured data, an analytics pipeline shows how Zepto would model outcomes end to end, and a support assistant shows how Zepto would put a grounded GenAI service in front of its own policies.

---

## 1. Setup

**Dependency approach: one `requirements.txt` per module**, not a single consolidated file. Each module has different, largely non-overlapping dependencies (web scraping vs. ML/data-science vs. RAG/API stack), so installing per-module keeps each environment lighter and avoids version conflicts between, say, `chromadb` and `imbalanced-learn`.

- `/requirements.txt` (repo root) → dependencies for **Module 1**
- `/analytics/requirements.txt` → dependencies for **Module 2**
- `/support_assistant/requirements.txt` → dependencies for **Module 3**

Create one virtual environment and install what you need for the module you're running, for example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Module 1
pip install -r requirements.txt

# Module 2
pip install -r analytics/requirements.txt

# Module 3
pip install -r support_assistant/requirements.txt
```

(On macOS/Linux, activate with `source .venv/bin/activate` instead.)

---

## 2. How to run each module

### Module 1 — Data Pipeline
```powershell
pip install -r requirements.txt
cd data_pipeline
python run_pipeline.py
```
This scrapes BooksToScrape live, cleans and converts the data, (re)builds `data/books.db` from scratch, runs the required SQL queries into `sql_outputs/`, and verifies the SQL JOIN against a `pandas.merge()` equivalent. See [`data_pipeline/README.md`](./data_pipeline/README.md) for full details and sample output.

### Module 2 — Analytics Pipeline
```powershell
pip install -r analytics/requirements.txt
cd analytics
python module2.py
```
The Titanic dataset is loaded once (via Seaborn, with `analytics/titanic.csv` committed as an offline fallback), cleaned, profiled, visualized, and then used for classification, imbalance handling, hyperparameter tuning, and a separate fare-regression task. All charts, the model comparison table, and the saved pipeline artifact are written to `analytics/outputs/`. See [`analytics/README.md`](./analytics/README.md) for the full write-up, metrics, and interpretations.

### Module 3 — Support Assistant
```powershell
pip install -r support_assistant/requirements.txt
cd support_assistant
python ingest.py          # embeds the 8 policy docs into ChromaDB
uvicorn api:app --reload  # starts the FastAPI service on http://127.0.0.1:8000
```
By default `MOCK_LLM=1`, so the whole pipeline (intent classification, retrieval, answer generation) runs deterministically with no LLM API key required. Swagger docs are at `http://127.0.0.1:8000/docs`. A Dockerfile is also included — see [`support_assistant/README.md`](./support_assistant/README.md) for build/run commands, the architecture write-up, and example request/response transcripts.

---

## 3. Design decisions — summary

**Module 1 (Data Pipeline).** Books are scraped from the first three available BooksToScrape categories (Travel, Mystery, Historical Fiction), yielding 61 rows — comfortably over the 60-book / 3-category minimum. Price and rating are parsed with `pd.to_numeric(..., errors="coerce")` rather than raising on bad input, so a malformed row degrades to a null value instead of crashing the pipeline. `price_inr` is computed from the fixed project-defined rate of **1 GBP = 105.50 INR** — a constant, not a live lookup. The schema is two tables (`categories`, `books`) linked by `category_id` as a primary/foreign key pair, which is the minimum normalized structure the task calls for. Full detail, including the exact SQL queries and sample output, is in the module README.

**Module 2 (Analytics Pipeline).** Missing values are handled per the assignment's percentage-threshold rule (drop rows under 5% missing, impute between 5–30%, drop the column above 30% when imputation would be unreliable) — this produced a cleaned 889-row, 14-column dataset. Modeling preprocessing is deliberately kept separate from the EDA-stage cleaning: it's implemented as a `ColumnTransformer` (median-impute + `StandardScaler` for numeric features, most-frequent-impute + one-hot for categorical features) wrapped inside each model's `Pipeline`, so it structurally cannot be fit on anything but the training split. Logistic Regression is the final recommended classifier based on F1 and AUC. Full metrics, interpretations, and the final recommendation are in the module README.

**Module 3 (Support Assistant).** Retrieval always runs for real in both modes (embeddings and ChromaDB need no API key), while only the final *generation* step branches on `MOCK_LLM`. The graded baseline (`MOCK_LLM=1`, the default) is fully deterministic: intent is classified by keyword heuristic and answers are template-filled from the top retrieved chunk, so the module needs zero external API calls to be graded end to end. The optional `MOCK_LLM=0` path adds a real Groq LLM call with Pydantic-schema retry-on-failure logic, purely as an ungraded stretch. Full architecture walkthrough is in the module README.

---

## 4. Repository structure

```text
zepto-data-ai-platform/
├── README.md                  ← you are here
├── requirements.txt            ← Module 1 dependencies
├── data_pipeline/
│   ├── README.md
│   ├── run_pipeline.py
│   ├── data/
│   │   ├── books.db
│   │   └── books_clean.csv
│   └── sql_outputs/
├── analytics/
│   ├── README.md
│   ├── requirements.txt
│   ├── module2.py
│   ├── titanic.csv
│   └── outputs/
└── support_assistant/
    ├── readme.md
    ├── requirements.txt
    ├── Dockerfile
    ├── api.py
    ├── graph.py
    ├── ingest.py
    ├── retriever.py
    ├── prompt_template.py
    ├── schemas.py
    └── docs/
```

---

## 5. Git workflow

Documentation and dependency fixes for this submission were developed on a feature branch and merged back into `main` (see the repository's commit/merge history), per the project's required git workflow.
