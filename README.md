# Hallucination Watchdog 🔬

A real-time LLM hallucination detection agent built with **Google ADK + Gemini** and **Arize Phoenix**. Built for the [Google Cloud Rapid Agent Hackathon](https://rapid-agent.devpost.com) — Arize track.

## What it does

- **Evaluates** any (query, context, response) triple for hallucinations using Phoenix's `HallucinationEvaluator` with Gemini as the judge LLM
- **Traces** every evaluation as an OpenTelemetry span to Arize Phoenix Cloud
- **Self-introspects** via the Phoenix MCP server — the agent queries its own trace history at runtime to detect drift and degradation
- **Recommends** prompt rewrites, retrieval fixes, or model swaps based on observed patterns

## Architecture

```
User / API call
      │
      ▼
┌─────────────────────────┐
│   FastAPI server        │  Cloud Run
│   (server.py)           │
└────────────┬────────────┘
             │ ADK agent.run()
             ▼
┌─────────────────────────┐
│  Gemini ADK Agent       │  gemini-2.5-flash
│  watchdog_agent.py      │
│                         │
│  Tools:                 │
│  • evaluate_response    │──► HallucinationEvaluator
│  • check_drift          │──► Phoenix MCP Server
│  • get_worst_offenders  │──► Phoenix MCP Server
└────────────┬────────────┘
             │ OTel spans
             ▼
┌─────────────────────────┐
│  Arize Phoenix Cloud    │  app.phoenix.arize.com
│  • Traces + spans       │
│  • Eval scores          │
│  • Drift dashboards     │
└─────────────────────────┘
```

## Quickstart

### 1. Get credentials

- **Phoenix Cloud**: free account at [app.phoenix.arize.com](https://app.phoenix.arize.com) → grab API key
- **Google Cloud**: project with Vertex AI enabled, or a Gemini API key

### 2. Set up environment

```bash
cp .env.example .env
# Fill in PHOENIX_API_KEY and GOOGLE_API_KEY
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
npm install -g @arizeai/phoenix-mcp@latest   # for MCP self-introspection
```

### 4. Run locally

```bash
source .env  # or: export $(cat .env | xargs)
python server.py
```

### 5. Test it

```bash
# Evaluate a single response
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the boiling point of water?",
    "context": "Water boils at 100°C at 1 atm pressure.",
    "response": "Water boils at 75°C at sea level.",
    "source_label": "my-model"
  }'

# Check for drift across recent evals
curl -X POST http://localhost:8080/drift \
  -H "Content-Type: application/json" \
  -d '{"project_name": "hallucination-watchdog", "lookback_n": 50}'

# Full agent endpoint (natural language)
curl -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Check hallucination drift and tell me which spans to investigate"}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

### 7. Deploy to Cloud Run

```bash
# Store secrets in Secret Manager first
echo -n "your-phoenix-key" | gcloud secrets create phoenix-api-key --data-file=-
echo -n "your-google-key"  | gcloud secrets create google-api-key --data-file=-

chmod +x deploy.sh
./deploy.sh
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/evaluate` | Score a single (query, context, response) triple |
| POST | `/drift` | Compute hallucination rate + trend from Phoenix traces |
| POST | `/worst-offenders` | Return top-K highest-scoring hallucination spans |
| POST | `/agent` | Full natural-language agent interface |

## Project structure

```
hallucination-watchdog/
├── agent/
│   └── watchdog_agent.py       # ADK agent definition + tool wiring
├── evaluator/
│   └── hallucination_eval.py   # Phoenix HallucinationEvaluator wrapper
├── mcp_loop/
│   └── phoenix_introspection.py # Phoenix MCP self-introspection loop
├── tests/
│   └── test_watchdog.py        # Pytest suite
├── server.py                   # FastAPI + Cloud Run entrypoint
├── Dockerfile                  # Cloud Run image
├── requirements.txt
├── deploy.sh                   # One-command Cloud Run deploy
└── .env.example
```

## Key design decisions

**Why async + ThreadPoolExecutor for evals?**
Phoenix's `HallucinationEvaluator` is synchronous and blocking. Running it in an executor lets the FastAPI event loop stay responsive under concurrent requests.

**Why MCP subprocess for introspection?**
The `@arizeai/phoenix-mcp` server exposes Phoenix's full trace query API over the MCP wire protocol, which is exactly what the hackathon judges expect to see. The REST fallback exists so tests don't require `npx`.

**Why Gemini as judge LLM?**
Using Gemini as both the agent model and the evaluator keeps the stack consistent. You can swap in GPT-4o or Claude by changing `EVAL_MODEL`.