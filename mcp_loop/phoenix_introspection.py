"""
Phoenix MCP Self-Introspection Loop
------------------------------------
Uses the Phoenix MCP server (@arizeai/phoenix-mcp) to let the agent query
its own trace history at runtime. This is the "self-improvement" feature
that Arize specifically rewards in judging.

The MCP server is launched as a subprocess (via npx) and called using the
MCP client protocol. All calls are async.
"""

import asyncio
import json
import os
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PHOENIX_BASE_URL = os.environ.get("PHOENIX_BASE_URL", "https://app.phoenix.arize.com")
PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY", "")

# ---------------------------------------------------------------------------
# Thin async Phoenix REST client (used when MCP server isn't available locally)
# The MCP server exposes the same data; this is the fallback + test path.
# ---------------------------------------------------------------------------

class PhoenixClient:
    """Minimal async wrapper around Phoenix Cloud REST API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=PHOENIX_BASE_URL,
            headers={
                "Authorization": f"Bearer {PHOENIX_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def get_spans(
        self,
        project_name: str,
        limit: int = 50,
        filter_attr: str | None = None,
        filter_value: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch spans from a Phoenix project.
        Optionally filter by a span attribute (e.g. eval.verdict=HALLUCINATION).
        """
        params: dict[str, Any] = {"project_name": project_name, "limit": limit}
        if filter_attr and filter_value:
            params["filter"] = f'{filter_attr} == "{filter_value}"'

        resp = await self._client.get("/v1/spans", params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def aclose(self) -> None:
        await self._client.aclose()


_phoenix_client: PhoenixClient | None = None


def _get_client() -> PhoenixClient:
    global _phoenix_client
    if _phoenix_client is None:
        _phoenix_client = PhoenixClient()
    return _phoenix_client


# ---------------------------------------------------------------------------
# MCP subprocess launcher — the proper hackathon integration
# ---------------------------------------------------------------------------

class PhoenixMCPSession:
    """
    Manages a long-lived npx @arizeai/phoenix-mcp subprocess.
    Communicates over JSON-RPC via stdio (MCP wire protocol).
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._request_id = 0

    async def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        async with self._lock:
            if self._proc is not None:
                return
            self._proc = await asyncio.create_subprocess_exec(
                "npx", "-y", "@arizeai/phoenix-mcp@latest",
                "--baseUrl", PHOENIX_BASE_URL,
                "--apiKey", PHOENIX_API_KEY,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info("Phoenix MCP server started (pid=%s)", self._proc.pid)

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Send a tools/call JSON-RPC request to the MCP server."""
        await self._ensure_started()
        assert self._proc and self._proc.stdin and self._proc.stdout

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        payload = (json.dumps(request) + "\n").encode()
        self._proc.stdin.write(payload)
        await self._proc.stdin.drain()

        raw = await self._proc.stdout.readline()
        response = json.loads(raw)

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        # Extract text content from MCP tool result
        content_blocks = response.get("result", {}).get("content", [])
        text_blocks = [b["text"] for b in content_blocks if b.get("type") == "text"]
        combined = "\n".join(text_blocks)

        try:
            return json.loads(combined)
        except json.JSONDecodeError:
            return combined

    async def aclose(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()


_mcp_session: PhoenixMCPSession | None = None


def _get_mcp_session() -> PhoenixMCPSession:
    global _mcp_session
    if _mcp_session is None:
        _mcp_session = PhoenixMCPSession()
    return _mcp_session


# ---------------------------------------------------------------------------
# High-level introspection functions (called by ADK tools in watchdog_agent.py)
# ---------------------------------------------------------------------------

async def get_drift_report(project_name: str, lookback_n: int = 50) -> dict:
    """
    Query Phoenix (via MCP) for the last N evaluation spans.
    Compute hallucination rate and trend direction.
    """
    try:
        # Prefer MCP server; fall back to REST if npx unavailable
        mcp = _get_mcp_session()
        raw = await mcp.call_tool(
            "get_spans",
            {"project_name": project_name, "limit": lookback_n},
        )
        spans = raw if isinstance(raw, list) else raw.get("spans", [])
    except Exception as exc:
        logger.warning("MCP unavailable (%s), falling back to REST", exc)
        client = _get_client()
        spans = await client.get_spans(project_name, limit=lookback_n)

    if not spans:
        return {
            "hallucination_rate": 0.0,
            "trend": "stable",
            "flagged_spans": [],
            "recommendation": "No evaluation data yet. Run some evaluations first.",
            "total_evaluated": 0,
        }

    verdicts = _extract_verdicts(spans)
    hallucination_rate = _compute_rate(verdicts)
    trend = _compute_trend(verdicts)
    flagged = _extract_flagged_span_ids(spans)
    recommendation = _generate_recommendation(hallucination_rate, trend, flagged)

    return {
        "hallucination_rate": round(hallucination_rate, 4),
        "trend": trend,
        "flagged_spans": flagged[:10],  # top 10 for brevity
        "recommendation": recommendation,
        "total_evaluated": len(verdicts),
    }


async def get_worst_spans(project_name: str, top_k: int = 5) -> dict:
    """
    Return the top-K spans with the highest hallucination scores.
    """
    try:
        mcp = _get_mcp_session()
        raw = await mcp.call_tool(
            "get_spans",
            {"project_name": project_name, "limit": 200},
        )
        spans = raw if isinstance(raw, list) else raw.get("spans", [])
    except Exception as exc:
        logger.warning("MCP unavailable (%s), falling back to REST", exc)
        client = _get_client()
        spans = await client.get_spans(
            project_name, limit=200,
            filter_attr="eval.verdict", filter_value="HALLUCINATION",
        )

    scored = []
    for span in spans:
        attrs = _get_attrs(span)
        verdict = attrs.get("eval.verdict", "")
        score = float(attrs.get("eval.score", 0.0))
        if verdict == "HALLUCINATION":
            scored.append({
                "span_id": attrs.get("eval.span_id", span.get("context", {}).get("span_id", "?")),
                "score": score,
                "query": attrs.get("eval.query", "")[:200],
                "explanation": attrs.get("eval.explanation", "")[:300],
                "source_label": attrs.get("eval.source_label", "unknown"),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"worst_offenders": scored[:top_k], "total_hallucinations_found": len(scored)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_attrs(span: dict) -> dict:
    """Normalise span attribute access across Phoenix REST and MCP shapes."""
    # REST API nests under "attributes"; MCP may return flat
    return span.get("attributes", span)


def _extract_verdicts(spans: list[dict]) -> list[str]:
    return [
        str(_get_attrs(s).get("eval.verdict", "UNCERTAIN")).upper()
        for s in spans
        if "eval.verdict" in _get_attrs(s)
    ]


def _compute_rate(verdicts: list[str]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v == "HALLUCINATION") / len(verdicts)


def _compute_trend(verdicts: list[str]) -> str:
    """Compare first half vs second half hallucination rate to detect trend."""
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


def _extract_flagged_span_ids(spans: list[dict]) -> list[str]:
    flagged = []
    for span in spans:
        attrs = _get_attrs(span)
        if str(attrs.get("eval.verdict", "")).upper() == "HALLUCINATION":
            sid = attrs.get("eval.span_id") or span.get("context", {}).get("span_id", "?")
            flagged.append(str(sid))
    return flagged


def _generate_recommendation(rate: float, trend: str, flagged: list[str]) -> str:
    if rate == 0.0:
        return "✅ No hallucinations detected. System looks healthy."
    if rate < 0.1 and trend != "degrading":
        return f"🟡 Low hallucination rate ({rate:.1%}). Monitor for increases."
    if rate >= 0.3 or trend == "degrading":
        return (
            f"🔴 High hallucination rate ({rate:.1%}, trend: {trend}). "
            f"Inspect flagged spans: {', '.join(flagged[:3])}. "
            "Consider: (1) improving retrieval context quality, "
            "(2) adding explicit 'answer only from context' instructions, "
            "(3) switching to a more grounded model variant."
        )
    return (
        f"🟠 Moderate hallucination rate ({rate:.1%}, trend: {trend}). "
        "Review flagged spans and consider prompt hardening."
    )
