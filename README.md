# Hallucination Watchdog 🔬

A real-time LLM hallucination detection agent built with **Google ADK + Gemini 2.5 Flash** and **Arize Phoenix**. Built for the [Google Cloud Rapid Agent Hackathon](https://rapid-agent.devpost.com) — Arize track.

🌐 **Live demo**: `https://hallucination-watchdog-375037039776.us-central1.run.app`

---

## What it does

- **Evaluates** any (query, context, response) triple for hallucinations using Arize Phoenix's `FaithfulnessEvaluator` with Gemini as the judge LLM
- **Traces** every evaluation as an OpenTelemetry span to Arize Phoenix Cloud
- **Detects drift** — monitors hallucination rate over time and flags when quality is degrading
- **Self-improves** — the agent reads its own evaluation history at runtime to surface patterns and recommend fixes
- **Natural language interface** — a single `/agent` endpoint accepts free-form requests and lets Gemini decide which tools to call

---

## Architecture

```
User / API call
      │
      ▼
┌─────────────────────────┐
│   FastAPI server        │  Google Cloud Run · us-central1
│   (server.py)           │
└────────────┬────────────┘
             │ ADK Runner
             ▼
┌─────────────────────────┐
│  Gemini ADK Agent       │  gemini-2.5-flash
│  watchdog_agent.py      │
│                         │
│  Tools:                 │
│  • evaluate_response    │──► FaithfulnessEvaluator (Arize Phoenix)
│  • check_drift          │──► eval_results.json (local history)
│  • get_worst_offenders  │──► eval_results.json (local history)
└────────────┬────────────┘
             │ OTel spans (OTLP)
             ▼
┌─────────────────────────┐
│  Arize Phoenix Cloud    │  app.phoenix.arize.com
│  • Traces + spans       │
│  • Eval scores          │
│  • Drift dashboards     │
└─────────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | Google ADK |
| LLM | Gemini 2.5 Flash via Vertex AI |
| Hallucination evaluation | Arize Phoenix FaithfulnessEvaluator |
| Observability | OpenTelemetry + Arize Phoenix Cloud |
| API server | FastAPI + uvicorn |
| Deployment | Google Cloud Run |
| Container registry | Google Artifact Registry |
| Language | Python 3.12 |

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI enabled
- Arize Phoenix Cloud account (free at [app.phoenix.arize.com](https://app.phoenix.arize.com))
- Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com))

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your_google_api_key"
$env:GOOGLE_GENAI_USE_VERTEXAI="False"   # True if using Vertex AI
$env:GOOGLE_CLOUD_PROJECT="your_gcp_project"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
$env:PHOENIX_API_KEY="your_phoenix_api_key"
$env:PHOENIX_BASE_URL="https://app.phoenix.arize.com"
$env:PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com"
$env:EVAL_MODEL="gemini-2.5-flash"

# Linux/Mac
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_GENAI_USE_VERTEXAI="False"
export PHOENIX_API_KEY="your_phoenix_api_key"
export PHOENIX_BASE_URL="https://app.phoenix.arize.com"
export PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com"
export EVAL_MODEL="gemini-2.5-flash"
```

### 4. Run locally

```bash
python server.py
```

Wait for:
```
INFO: Watchdog agent ready
INFO: Uvicorn running on http://0.0.0.0:8080
```

### 5. Test it

```bash
# Health check
curl http://localhost:8080/health

# Evaluate a hallucinated response
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "context": "The capital of France is Paris.",
    "response": "The capital of France is London.",
    "source_label": "my-model"
  }'

# Check drift across recent evaluations
curl -X POST http://localhost:8080/drift \
  -H "Content-Type: application/json" \
  -d '{"project_name": "hallucination-watchdog", "lookback_n": 50}'

# Full agent — natural language interface
curl -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Evaluate this: query=What causes rain?, context=Rain is caused by water vapor condensing in clouds., response=Rain is caused by the moon pulling water from the ocean. Then check drift and tell me what to fix."}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/evaluate` | Score a single (query, context, response) triple |
| POST | `/drift` | Compute hallucination rate + trend from evaluation history |
| POST | `/worst-offenders` | Return top-K highest-scoring hallucination spans |
| POST | `/agent` | Full natural-language agent interface |

### Example response — `/evaluate`

```json
{
  "verdict": "HALLUCINATION",
  "score": 0.5,
  "explanation": "The context states the capital is Paris, but the response claims it is London — a direct contradiction.",
  "span_id": "93409bab-b48c-4c2f-8cb9-c4dec284a592",
  "latency_ms": 1718.32,
  "source_label": "my-model"
}
```

### Example response — `/drift`

```json
{
  "hallucination_rate": 0.6667,
  "trend": "stable",
  "flagged_spans": ["What is the boiling point of water?", "What is the capital of France?"],
  "recommendation": "🔴 High hallucination rate (66.7%). Consider: (1) improving retrieval context quality, (2) adding explicit 'answer only from context' instructions, (3) switching to a more grounded model variant.",
  "total_evaluated": 3
}
```

---

## Deploy to Cloud Run

### 1. Build and push Docker image

```bash
docker build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/YOUR_PROJECT/hallucination-watchdog/watchdog:latest .

docker push us-central1-docker.pkg.dev/YOUR_PROJECT/hallucination-watchdog/watchdog:latest
```

### 2. Deploy

```bash
gcloud run deploy hallucination-watchdog \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT/hallucination-watchdog/watchdog:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --min-instances 1 \
  --set-env-vars "GOOGLE_API_KEY=your_key,PHOENIX_API_KEY=your_key,GOOGLE_GENAI_USE_VERTEXAI=False,EVAL_MODEL=gemini-2.5-flash,PHOENIX_BASE_URL=https://app.phoenix.arize.com,PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com"
```

---

## Project structure

```
hallucination-watchdog/
├── agent/
│   └── watchdog_agent.py          # ADK agent + tool definitions + OTel tracing
├── evaluator/
│   └── hallucination_eval.py      # Arize Phoenix FaithfulnessEvaluator wrapper
├── mcp_loop/
│   └── phoenix_introspection.py   # Drift detection + self-improvement loop
├── tests/
│   └── test_watchdog.py           # Pytest suite
├── server.py                      # FastAPI + Cloud Run entrypoint
├── Dockerfile                     # Cloud Run optimised image
├── requirements.txt
├── .env.example                   # Environment variable template
└── LICENSE                        # MIT
```

---

## Key design decisions

**Why `FaithfulnessEvaluator` instead of `HallucinationEvaluator`?**
`HallucinationEvaluator` was deprecated in recent Phoenix versions. `FaithfulnessEvaluator` is the recommended replacement with `faithful`/`unfaithful` labels which we map to `FACTUAL`/`HALLUCINATION` for consistency.

**Why `ThreadPoolExecutor` for evaluations?**
Phoenix's evaluator is synchronous and blocking. Wrapping it in a thread executor keeps FastAPI's async event loop responsive under concurrent requests without degrading throughput.

**Why local JSON for drift instead of Phoenix REST API?**
Phoenix Cloud's REST API requires account-specific base URLs and auth headers that vary by deployment. Local JSON tracking is reliable, portable, and works identically in local dev and Cloud Run.

**Why `--min-instances 1` on Cloud Run?**
The ADK agent and Phoenix evaluator have significant cold start overhead (~30s). Keeping one instance always warm ensures consistent demo and production latency.

---

## License

MIT — see [LICENSE](LICENSE)