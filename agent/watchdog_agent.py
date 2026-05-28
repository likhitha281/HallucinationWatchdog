"""
Hallucination Watchdog Agent
-----------------------------
A Google ADK agent that:
  1. Accepts (query, context, response) triples
  2. Scores them via Arize Phoenix HallucinationEvaluator
  3. Logs every evaluation as a span to Phoenix Cloud
  4. Exposes a self-introspection tool via Phoenix MCP to surface drift
"""

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import phoenix as px
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini

from evaluator.hallucination_eval import HallucinationEvalResult, run_hallucination_eval
from mcp_loop.phoenix_introspection import get_drift_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap Phoenix tracing (call once at startup)
# ---------------------------------------------------------------------------

def init_tracing() -> None:
    """Wire OpenTelemetry → Phoenix Cloud via OTLP."""
    phoenix_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com")
    api_key = os.environ.get("PHOENIX_API_KEY", "")

    exporter = OTLPSpanExporter(
        endpoint=f"{phoenix_endpoint}/v1/traces",
        headers={"api_key": api_key},
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument Google ADK + google-genai SDK calls
    GoogleADKInstrumentor().instrument()
    GoogleGenAIInstrumentor().instrument()

    logger.info("Phoenix tracing initialised → %s", phoenix_endpoint)


# ---------------------------------------------------------------------------
# ADK Tool definitions (what the agent can call)
# ---------------------------------------------------------------------------

@dataclass
class EvalRequest:
    query: str
    context: str        # retrieved docs / ground-truth reference
    response: str       # LLM output under scrutiny
    source_label: str = "unknown"  # optional tag (e.g. "gpt-4o", "gemini-flash")


async def evaluate_response(
    query: str,
    context: str,
    response: str,
    source_label: str = "unknown",
) -> dict:
    """
    ADK tool: score a single (query, context, response) triple for hallucinations.

    Returns a dict with:
      - verdict:     HALLUCINATION | FACTUAL | UNCERTAIN
      - score:       float 0-1 (1 = definitely hallucinating)
      - explanation: string reasoning from the judge LLM
      - span_id:     Phoenix trace span ID for downstream lookup
    """
    req = EvalRequest(query=query, context=context, response=response, source_label=source_label)
    result: HallucinationEvalResult = await run_hallucination_eval(req)
    return result.to_dict()


async def check_hallucination_drift(
    project_name: str = "hallucination-watchdog",
    lookback_n: int = 50,
) -> dict:
    """
    ADK tool: query Phoenix MCP for the last N evaluations and compute drift metrics.

    Returns:
      - hallucination_rate: float (0-1)
      - trend:              "improving" | "degrading" | "stable"
      - flagged_spans:      list of span IDs with HALLUCINATION verdict
      - recommendation:     actionable string the agent surfaces to the user
    """
    return await get_drift_report(project_name=project_name, lookback_n=lookback_n)


async def get_worst_offenders(
    project_name: str = "hallucination-watchdog",
    top_k: int = 5,
) -> dict:
    """
    ADK tool: return the top-K highest-scoring hallucination spans from Phoenix,
    including their query, context snippet, and offending response.
    Useful for targeted prompt debugging.
    """
    from mcp_loop.phoenix_introspection import get_worst_spans
    return await get_worst_spans(project_name=project_name, top_k=top_k)


# ---------------------------------------------------------------------------
# Build the ADK agent
# ---------------------------------------------------------------------------

def build_watchdog_agent() -> Agent:
    system_prompt = """
You are the Hallucination Watchdog — an expert AI quality-assurance agent.

Your job is to:
1. Evaluate LLM responses for factual grounding using the `evaluate_response` tool.
2. Monitor trends over time using `check_hallucination_drift`.
3. Surface the most problematic outputs using `get_worst_offenders`.
4. Give concrete, actionable recommendations: prompt rewrites, retrieval fixes, or model swap suggestions.

Guidelines:
- Always call `evaluate_response` when given a (query, context, response) triple.
- After 10+ evaluations, proactively call `check_hallucination_drift` to surface trends.
- When hallucination_rate > 0.3, call `get_worst_offenders` and recommend a fix.
- Be precise: cite span IDs when referencing specific failures.
- Never guess — base every claim on tool output.
"""

    agent = Agent(
        name="hallucination_watchdog",
        model=Gemini(model="gemini-2.5-flash"),
        description="Scores LLM responses for hallucinations and monitors quality drift via Arize Phoenix.",
        instruction=system_prompt,
        tools=[
            FunctionTool(evaluate_response),
            FunctionTool(check_hallucination_drift),
            FunctionTool(get_worst_offenders),
        ],
    )
    return agent


# ---------------------------------------------------------------------------
# Entrypoint (local dev / Cloud Run)
# ---------------------------------------------------------------------------

async def main() -> None:
    init_tracing()
    agent = build_watchdog_agent()

    # Example: evaluate a single response
    response = await agent.run(
        user_message=(
            "Evaluate this response: "
            "Query: 'What is the boiling point of water at sea level?' "
            "Context: 'Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm).' "
            "Response: 'Water boils at 90°C at sea level.' "
            "Then check for drift across recent evaluations."
        )
    )
    print(response.text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
