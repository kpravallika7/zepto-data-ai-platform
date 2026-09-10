# Zepto Support Assistant — Module 3

A Retrieval-Augmented Generation (RAG) support assistant for Zepto policy questions. The system ingests support documents, embeds them into ChromaDB, retrieves relevant context, and generates a structured answer through a LangGraph workflow.

## Overview

This module implements:

- Document ingestion and chunking
- SentenceTransformer embeddings
- Persistent ChromaDB vector storage
- Semantic retrieval
- Prompt templating with grounding constraints and a few-shot example
- LangGraph intent classification and conditional routing
- Deterministic mock mode for the graded baseline
- Optional real LLM mode using Groq
- Pydantic response validation
- Retry-on-failure logic for real LLM responses
- FastAPI REST API
- Docker containerization

The default `MOCK_LLM=1` mode requires no external LLM API key.

---

## Architecture

The pipeline follows this order:

```text
Ingestion → Embedding → Retrieval → Generation
```

### 1. Ingestion

**File:** `ingest.py`

The ingestion stage:

1. Loads all support documents from `docs/`
2. Creates one chunk per document
3. Generates embeddings using `all-MiniLM-L6-v2`
4. Stores documents, embeddings, IDs, and metadata in persistent ChromaDB

### 2. Embedding

**Components:** `ingest.py`, `retriever.py`

The SentenceTransformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

creates 384-dimensional embeddings for both stored documents and incoming queries.

### 3. Retrieval

**File:** `retriever.py`

**Component:** `retrieve_documents()`

The query is embedded and compared against the ChromaDB collection using cosine distance. The top matching documents are returned with their source IDs, text, distances, and similarity scores.

### 4. Generation

**File:** `graph.py`

**Nodes:**

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The generation stage branches depending on `MOCK_LLM`.

- `MOCK_LLM=1`: deterministic local/mock behavior; no external LLM API call.
- `MOCK_LLM=0`: optional real Groq LLM generation using `GROQ_API_KEY`.

---

## Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── retriever.py
├── prompt_template.py
├── schemas.py
├── graph.py
├── api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 1. Document Ingestion

Run:

```powershell
python ingest.py
```

Expected result:

```text
Documents loaded: 8
Chunks created: 8
Embedding shape: (8, 384)
ChromaDB collection: zepto_support
Stored records: 8
Ingestion completed successfully.
```

The ingestion process recreates the ChromaDB collection so repeated runs remain deterministic.

The collection name is:

```text
zepto_support
```

---

## 2. Retrieval Test

Run:

```powershell
python retriever.py
```

A delivery-policy query should retrieve the delivery policy as the top result.

Example:

```text
RETRIEVAL TEST

Result 1
Source: doc_01.txt
Similarity: 0.5034
Text: Delivery Policy: "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo...

Result 2
Source: doc_05.txt
Similarity: 0.3112
Text: Order Cancellation Policy: ...

Result 3
Source: doc_02.txt
Similarity: 0.2614
Text: Returns & Refunds: ...
```

---

## 3. Prompt Template

**File:** `prompt_template.py`

The support prompt contains the required components:

1. Role
2. Context
3. Task
4. Format
5. Length constraint

It also contains:

- A negative grounding constraint
- A few-shot example

The prompt explicitly instructs the assistant to:

- Answer only using the supplied context
- Avoid invention
- Avoid unsupported assumptions or inference
- State that there is insufficient context when the answer is not supported
- Keep the answer concise

The prompt is used by both retrieval and optional real-LLM generation.

---

## 4. LangGraph Workflow

**File:** `graph.py`

The graph contains three named nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
START
  │
  ▼
classify_intent
  │
  ├── policy_question ──► retrieve_and_answer ──► END
  │
  └── general_question ─► direct_answer ───────► END
```

### Intent classification

In default mock mode, keyword heuristics classify policy-related questions.

The following keywords route to `policy_question`:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Unrelated questions route to:

```text
general_question
```

No LLM call is made during intent classification in mock mode.

---

## 5. Mock LLM Mode

Mock mode is the default:

```text
MOCK_LLM=1
```

It is deterministic and requires no Groq API key.

Example policy query:

```text
How much is delivery?
```

Expected behavior:

```text
intent = policy_question
route = retrieve_and_answer
```

The mock retrieval response follows this format:

```text
Based on the retrieved context: <top retrieved chunk>
```

The sources are populated from the retrieved documents and confidence is deterministic.

Example general query:

```text
What is the capital of India?
```

Expected behavior:

```text
intent = general_question
route = direct_answer
```

The mock direct response is:

```text
I can only answer questions about Zepto policies right now.
```

No external LLM network call is made in mock mode.

---

## 6. Optional Real LLM Mode

The optional real-LLM extension is enabled with:

```text
MOCK_LLM=0
```

This uses Groq for generation and intent classification.

Set the API key as an environment variable rather than hardcoding it:

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Then run the application normally.

If `MOCK_LLM=0` is used without a valid `GROQ_API_KEY`, the real-LLM path will fail because the external model requires authentication.

The real-LLM response is validated with Pydantic. Invalid responses are retried with a corrective schema instruction, up to three total attempts.

---

## 7. Response Schema

**File:** `schemas.py`

The API response is validated using Pydantic:

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

The schema enforces:

- `answer` must be non-empty
- `sources` must be a list of strings
- `confidence` must be between `0.0` and `1.0`

Example:

```json
{
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo",
  "sources": ["doc_01.txt", "doc_05.txt", "doc_02.txt"],
  "confidence": 1.0
}
```

---

## 8. FastAPI

**File:** `api.py`

Start the API with:

```powershell
uvicorn api:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Health check

```text
GET /health
```

### Ask endpoint

```text
POST /ask
```

Request:

```json
{
  "query": "How much does delivery cost?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, dependingon the customer's delivery zone and current order vo",
  "sources": ["doc_01.txt", "doc_05.txt", "doc_02.txt"],
  "confidence": 1.0
}
```

For an unrelated query:

```json
{
  "query": "What is the capital of India?"
}
```

Example response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## 9. Docker

The project includes a `Dockerfile` for local container execution.

### Build

From the `support_assistant` directory:

```powershell
docker build -t zepto-support-assistant .
```

If Docker is being accessed through WSL:

```powershell
wsl -d Ubuntu -- docker build -t zepto-support-assistant .
```

### Run

```powershell
docker run --rm -p 8000:7860 -e MOCK_LLM=1 zepto-support-assistant
```

If using WSL:

```powershell
wsl -d Ubuntu -- docker run --rm -p 8000:7860 -e MOCK_LLM=1 zepto-support-assistant
```

The container exposes port:

```text
7860
```

and maps it to local port:

```text
8000
```

The API can then be accessed at:

```text
http://127.0.0.1:8000
```

The container was successfully built and run locally with the default mock mode.

---

## 10. Requirements

The runtime dependencies are listed in:

```text
requirements.txt
```

Current direct dependencies:

```text
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.5
chromadb==1.5.9
sentence-transformers==6.0.1
langgraph==1.2.11
```

---

## 11. Verification

The following Module 3 checks have been completed:

### Ingestion

- 8 support documents loaded
- 8 chunks created
- 384-dimensional embeddings generated
- 8 records stored in ChromaDB

### Retrieval

- ChromaDB retrieval works
- Delivery query returns `doc_01.txt` as the top source

### Prompt

- All five required skeleton components are present
- Negative grounding constraint is present
- Few-shot example is present

### LangGraph

- Three named nodes are present
- Conditional routing works
- Policy queries use retrieval
- General queries use the direct path

### Mock mode

- `MOCK_LLM=1` is the default
- No external LLM API call is required
- Policy and general queries produce deterministic outputs
- Response schema is Pydantic validated

### API

- FastAPI starts successfully
- `/health` works
- `/ask` works for both policy and unrelated queries
- Swagger documentation is available

### Docker

- Docker image builds successfully
- Container starts successfully
- API is accessible locally from the container
