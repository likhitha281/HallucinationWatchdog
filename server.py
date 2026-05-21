"""
FastAPI server — Cloud Run entrypoint
--------------------------------------
Exposes the Hallucination Watchdog as a REST API.
Designed for Cloud Run: stateless, fast cold starts, PORT env var.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.watchdog_agent import build_watchdog_agent, init_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    app.state.agent = build_watchdog_agent()
    logger.info("Watchdog agent ready")
    yield


app = FastAPI(
    title="Hallucination Watchdog",
    description="Real-time LLM hallucination detection powered by Gemini + Arize Phoenix",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    query: str = Field(..., description="The user query sent to the LLM")
    context: str = Field(..., description="Reference context / retrieved documents")
    response: str = Field(..., description="The LLM response to evaluate")
    source_label: str = Field("unknown", description="Model or system tag (e.g. 'gpt-4o')")


class DriftRequest(BaseModel):
    project_name: str = Field("hallucination-watchdog")
    lookback_n: int = Field(50, ge=5, le=500)


class AgentRequest(BaseModel):
    message: str = Field(..., description="Free-form message to the watchdog agent")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(req: EvalRequest) -> dict:
    from evaluator.hallucination_eval import EvalRequest as InternalReq, run_hallucination_eval
    from mcp_loop.phoenix_introspection import _save_eval_result

    internal = InternalReq(
        query=req.query,
        context=req.context,
        response=req.response,
        source_label=req.source_label,
    )
    try:
        result = await run_hallucination_eval(internal)
        _save_eval_result(result.verdict, result.score, req.query, req.source_label)
        return result.to_dict()
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/drift")
async def drift(req: DriftRequest) -> dict:
    from mcp_loop.phoenix_introspection import get_drift_report

    try:
        return await get_drift_report(
            project_name=req.project_name,
            lookback_n=req.lookback_n,
        )
    except Exception as exc:
        logger.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/worst-offenders")
async def worst_offenders(project_name: str = "hallucination-watchdog", top_k: int = 5) -> dict:
    from mcp_loop.phoenix_introspection import get_worst_spans

    try:
        return await get_worst_spans(project_name=project_name, top_k=top_k)
    except Exception as exc:
        logger.exception("Worst-offender lookup failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent")
async def agent_chat(req: AgentRequest) -> dict:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    agent = app.state.agent
    try:
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="watchdog",
            session_service=session_service,
        )
        session = await session_service.create_session(
            app_name="watchdog",
            user_id="user",
        )

        content = Content(role="user", parts=[Part(text=req.message)])
        response_text = ""
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        response_text += part.text

        return {"response": response_text}
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, log_level="info")