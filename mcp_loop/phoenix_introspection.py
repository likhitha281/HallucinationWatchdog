"""
Phoenix MCP Self-Introspection Loop
-------------------------------------
Tracks evaluation results locally and computes drift metrics.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_FILE = "eval_results.json"


# ---------------------------------------------------------------------------
# Local result storage
# ---------------------------------------------------------------------------

def _save_eval_result(verdict: str, score: float, query: str, source_label: str) -> None:
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)

    results.append({
        "verdict": verdict,
        "score": score,
        "query": query[:200],
        "source_label": source_label,
        "timestamp": datetime.utcnow().isoformat(),
    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

async def get_drift_report(project_name: str, lookback_n: int = 50) -> dict:
    if not os.path.exists(RESULTS_FILE):
        return {
            "hallucination_rate": 0.0,
            "trend": "stable",
            "flagged_spans": [],
            "recommendation": "No evaluation data yet. Run some evaluations first.",
            "total_evaluated": 0,
        }

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    results = results[-lookback_n:]
    verdicts = [r["verdict"].upper() for r in results]
    hallucination_rate = _compute_rate(verdicts)
    trend = _compute_trend(verdicts)
    flagged = [r["query"] for r in results if r["verdict"].upper() == "HALLUCINATION"]
    recommendation = _generate_recommendation(hallucination_rate, trend, flagged)

    return {
        "hallucination_rate": round(hallucination_rate, 4),
        "trend": trend,
        "flagged_spans": flagged[:10],
        "recommendation": recommendation,
        "total_evaluated": len(verdicts),
    }


async def get_worst_spans(project_name: str, top_k: int = 5) -> dict:
    if not os.path.exists(RESULTS_FILE):
        return {"worst_offenders": [], "total_hallucinations_found": 0}

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    hallucinations = [r for r in results if r["verdict"].upper() == "HALLUCINATION"]
    hallucinations.sort(key=lambda x: x["score"], reverse=True)

    return {
        "worst_offenders": hallucinations[:top_k],
        "total_hallucinations_found": len(hallucinations),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rate(verdicts: list) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v == "HALLUCINATION") / len(verdicts)


def _compute_trend(verdicts: list) -> str:
    if len(verdicts) < 10:
        return "stable"
    mid = len(verdicts) // 2
    early_rate = _compute_rate(verdicts[:mid])
    recent_rate = _compute_rate(verdicts[mid:])
    delta = recent_rate - early_rate
    if delta > 0.05:
        return "degrading"
    elif delta < -0.05:
        return "improving"
    return "stable"


def _extract_flagged_span_ids(spans: list) -> list:
    return []


def _generate_recommendation(rate: float, trend: str, flagged: list) -> str:
    if rate == 0.0:
        return "✅ No hallucinations detected. System looks healthy."
    if rate < 0.1 and trend != "degrading":
        return f"🟡 Low hallucination rate ({rate:.1%}). Monitor for increases."
    if rate >= 0.3 or trend == "degrading":
        return (
            f"🔴 High hallucination rate ({rate:.1%}, trend: {trend}). "
            "Consider: (1) improving retrieval context quality, "
            "(2) adding explicit 'answer only from context' instructions, "
            "(3) switching to a more grounded model variant."
        )
    return (
        f"🟠 Moderate hallucination rate ({rate:.1%}, trend: {trend}). "
        "Review flagged spans and consider prompt hardening."
    )