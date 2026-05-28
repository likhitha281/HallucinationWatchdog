"""
Hallucination Evaluator
-----------------------
Uses Arize Phoenix FaithfulnessEvaluator (replaces deprecated HallucinationEvaluator).
Labels: 'faithful' (score=1.0) | 'unfaithful' (score=0.0)
We map these to FACTUAL | HALLUCINATION for consistency with the rest of the codebase.
"""

import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Literal

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from phoenix.evals.metrics import FaithfulnessEvaluator
from phoenix.evals.llm import LLM

_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("EVAL_CONCURRENCY", "8")))
_tracer = trace.get_tracer("hallucination-watchdog.evaluator")

Verdict = Literal["HALLUCINATION", "FACTUAL", "UNCERTAIN"]


@dataclass
class EvalRequest:
    query: str
    context: str
    response: str
    source_label: str = "unknown"


@dataclass
class HallucinationEvalResult:
    verdict: Verdict
    score: float
    explanation: str
    span_id: str
    latency_ms: float
    source_label: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_hallucination(self) -> bool:
        return self.verdict == "HALLUCINATION"


_evaluator: FaithfulnessEvaluator | None = None


def _get_evaluator() -> FaithfulnessEvaluator:
    global _evaluator
    if _evaluator is None:
        llm = LLM(
            model=os.environ.get("EVAL_MODEL", "gemini-2.5-flash"),
            provider="google",
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
        _evaluator = FaithfulnessEvaluator(llm=llm)
    return _evaluator


def _sync_evaluate(query: str, context: str, response: str) -> tuple[Verdict, float, str]:
    evaluator = _get_evaluator()
    results = evaluator.evaluate(
        eval_input={"input": query, "output": response, "context": context}
    )

    result_row = results[0] if results else None

    if result_row is None:
        return "UNCERTAIN", 0.5, "No result returned"

    # Get label — Score object has .label attribute
    label = str(getattr(result_row, "label", None) or "uncertain").lower()
    score = float(getattr(result_row, "score", 0.5) or 0.5)
    explanation = str(getattr(result_row, "explanation", "") or "")

    # Map faithful/unfaithful → FACTUAL/HALLUCINATION
    if label == "faithful":
        verdict: Verdict = "FACTUAL"
        score = 1.0 - score  # invert: faithful=1.0 → hallucination_score=0.0
    elif label == "unfaithful":
        verdict = "HALLUCINATION"
        score = 1.0 - score
    else:
        verdict = "UNCERTAIN"

    return verdict, score, explanation


async def run_hallucination_eval(req: EvalRequest) -> HallucinationEvalResult:
    loop = asyncio.get_event_loop()
    span_id = str(uuid.uuid4())

    with _tracer.start_as_current_span("hallucination_eval") as span:
        span.set_attribute("eval.query", req.query[:500])
        span.set_attribute("eval.context_length", len(req.context))
        span.set_attribute("eval.response_length", len(req.response))
        span.set_attribute("eval.source_label", req.source_label)
        span.set_attribute("eval.span_id", span_id)

        t0 = time.perf_counter()
        try:
            verdict, score, explanation = await loop.run_in_executor(
                _executor,
                _sync_evaluate,
                req.query,
                req.context,
                req.response,
            )
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        latency_ms = (time.perf_counter() - t0) * 1000

        span.set_attribute("eval.verdict", verdict)
        span.set_attribute("eval.score", score)
        span.set_attribute("eval.latency_ms", round(latency_ms, 2))
        span.set_attribute("eval.explanation", explanation[:1000])

        if verdict == "HALLUCINATION":
            span.set_status(Status(StatusCode.ERROR, "Hallucination detected"))
        else:
            span.set_status(Status(StatusCode.OK))

    return HallucinationEvalResult(
        verdict=verdict,
        score=score,
        explanation=explanation,
        span_id=span_id,
        latency_ms=round(latency_ms, 2),
        source_label=req.source_label,
    )