"""
Test suite for Hallucination Watchdog
--------------------------------------
Run with: pytest tests/ -v
Requires: PHOENIX_API_KEY set (or mocked), GOOGLE_API_KEY set
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def env_vars(monkeypatch):
    monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
    monkeypatch.setenv("PHOENIX_BASE_URL", "https://app.phoenix.arize.com")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


FACTUAL_TRIPLE = {
    "query": "What is the boiling point of water at sea level?",
    "context": "Water boils at 100°C (212°F) at 1 atm pressure.",
    "response": "Water boils at 100°C at sea level.",
    "source_label": "test-model",
}

HALLUCINATION_TRIPLE = {
    "query": "What is the boiling point of water at sea level?",
    "context": "Water boils at 100°C (212°F) at 1 atm pressure.",
    "response": "Water boils at 75°C at sea level.",  # wrong
    "source_label": "test-model",
}


# ---------------------------------------------------------------------------
# Evaluator unit tests
# ---------------------------------------------------------------------------

class TestHallucinationEval:

    @pytest.mark.asyncio
    async def test_factual_response_returns_factual_verdict(self):
        from evaluator.hallucination_eval import EvalRequest, run_hallucination_eval

        mock_result = {"label": "FACTUAL", "score": 0.05, "explanation": "Response matches context."}

        with patch("evaluator.hallucination_eval._sync_evaluate", return_value=("FACTUAL", 0.05, "Response matches context.")):
            req = EvalRequest(**FACTUAL_TRIPLE)
            result = await run_hallucination_eval(req)

        assert result.verdict == "FACTUAL"
        assert result.score < 0.5
        assert not result.is_hallucination

    @pytest.mark.asyncio
    async def test_hallucinated_response_detected(self):
        from evaluator.hallucination_eval import EvalRequest, run_hallucination_eval

        with patch("evaluator.hallucination_eval._sync_evaluate", return_value=("HALLUCINATION", 0.95, "Response contradicts context.")):
            req = EvalRequest(**HALLUCINATION_TRIPLE)
            result = await run_hallucination_eval(req)

        assert result.verdict == "HALLUCINATION"
        assert result.score > 0.5
        assert result.is_hallucination

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        from evaluator.hallucination_eval import EvalRequest, run_hallucination_eval

        with patch("evaluator.hallucination_eval._sync_evaluate", return_value=("FACTUAL", 0.1, "OK")):
            req = EvalRequest(**FACTUAL_TRIPLE)
            result = await run_hallucination_eval(req)

        d = result.to_dict()
        assert all(k in d for k in ["verdict", "score", "explanation", "span_id", "latency_ms", "source_label"])

    @pytest.mark.asyncio
    async def test_unknown_verdict_normalised_to_uncertain(self):
        from evaluator.hallucination_eval import EvalRequest, run_hallucination_eval

        with patch("evaluator.hallucination_eval._sync_evaluate", return_value=("UNKNOWN_LABEL", 0.5, "")):
            req = EvalRequest(**FACTUAL_TRIPLE)
            result = await run_hallucination_eval(req)

        assert result.verdict == "UNCERTAIN"


# ---------------------------------------------------------------------------
# Drift detection tests
# ---------------------------------------------------------------------------

class TestDriftDetection:

    def _make_spans(self, verdicts: list[str]) -> list[dict]:
        return [
            {"attributes": {"eval.verdict": v, "eval.score": 0.9 if v == "HALLUCINATION" else 0.1, "eval.span_id": str(i)}}
            for i, v in enumerate(verdicts)
        ]

    @pytest.mark.asyncio
    async def test_zero_hallucinations_returns_healthy(self):
        from mcp_loop.phoenix_introspection import get_drift_report

        spans = self._make_spans(["FACTUAL"] * 20)
        with patch("mcp_loop.phoenix_introspection._get_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get_spans.return_value = spans
            mock_client_factory.return_value = mock_client

            with patch("mcp_loop.phoenix_introspection._get_mcp_session", side_effect=RuntimeError("no npx")):
                report = await get_drift_report("test-project", lookback_n=20)

        assert report["hallucination_rate"] == 0.0
        assert "✅" in report["recommendation"]

    @pytest.mark.asyncio
    async def test_high_hallucination_rate_flagged(self):
        from mcp_loop.phoenix_introspection import get_drift_report

        spans = self._make_spans(["HALLUCINATION"] * 15 + ["FACTUAL"] * 5)
        with patch("mcp_loop.phoenix_introspection._get_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get_spans.return_value = spans
            mock_client_factory.return_value = mock_client

            with patch("mcp_loop.phoenix_introspection._get_mcp_session", side_effect=RuntimeError("no npx")):
                report = await get_drift_report("test-project", lookback_n=20)

        assert report["hallucination_rate"] >= 0.5
        assert "🔴" in report["recommendation"]

    @pytest.mark.asyncio
    async def test_degrading_trend_detected(self):
        from mcp_loop.phoenix_introspection import _compute_trend

        # First half: all factual; second half: all hallucination
        verdicts = ["FACTUAL"] * 10 + ["HALLUCINATION"] * 10
        assert _compute_trend(verdicts) == "degrading"

    @pytest.mark.asyncio
    async def test_improving_trend_detected(self):
        from mcp_loop.phoenix_introspection import _compute_trend

        verdicts = ["HALLUCINATION"] * 10 + ["FACTUAL"] * 10
        assert _compute_trend(verdicts) == "improving"

    @pytest.mark.asyncio
    async def test_empty_spans_returns_safe_default(self):
        from mcp_loop.phoenix_introspection import get_drift_report

        with patch("mcp_loop.phoenix_introspection._get_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get_spans.return_value = []
            mock_client_factory.return_value = mock_client

            with patch("mcp_loop.phoenix_introspection._get_mcp_session", side_effect=RuntimeError("no npx")):
                report = await get_drift_report("empty-project")

        assert report["total_evaluated"] == 0
        assert "No evaluation data" in report["recommendation"]


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------

class TestServerEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_evaluate_endpoint_returns_verdict(self, client):
        with patch("evaluator.hallucination_eval._sync_evaluate", return_value=("FACTUAL", 0.1, "Correct")):
            resp = client.post("/evaluate", json=FACTUAL_TRIPLE)

        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] in ("HALLUCINATION", "FACTUAL", "UNCERTAIN")
        assert "score" in body

    def test_evaluate_missing_fields_returns_422(self, client):
        resp = client.post("/evaluate", json={"query": "test"})  # missing context + response
        assert resp.status_code == 422
