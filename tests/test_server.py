"""MCP surface. Exercised through the real MCPServer, not a stand-in."""

from __future__ import annotations

import json

import pytest

from sota_anchor.catalog import build_catalog
from sota_anchor.retriever import EvidenceSet

from .conftest import NOW
from .test_arbiter import OBSOLETE, STILL_VALID, evidence_set
from .test_inversion import VALID, FakeLLM


@pytest.fixture
def server(monkeypatch, raw_models):
    from sota_anchor import server as module

    async def fake_fetch(**kwargs):
        return build_catalog(raw_models, now=NOW)

    monkeypatch.setattr(module, "fetch_catalog", fake_fetch)
    return module.build_server()


def text_of(result) -> str:
    return "\n".join(getattr(block, "text", "") for block in (result.content or []))


class TestSurface:
    async def test_server_is_named_for_the_project(self, server):
        assert server.name == "sota-anchor"

    async def test_exposes_the_check_what_exists_tool(self, server):
        assert "check_what_exists" in {tool.name for tool in await server.list_tools()}

    async def test_tool_takes_a_pitch_argument(self, server):
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        assert "pitch" in (tool.input_schema or {}).get("properties", {})

    async def test_tool_is_described_so_an_agent_knows_when_to_call_it(self, server):
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        assert tool.description and len(tool.description) > 40

    async def test_exposes_the_active_models_resource(self, server):
        assert "models://active" in {str(r.uri) for r in await server.list_resources()}

    async def test_resource_is_declared_as_json(self, server):
        resource = next(r for r in await server.list_resources() if str(r.uri) == "models://active")
        assert resource.mime_type == "application/json"

    async def test_exposes_the_init_project_prompt(self, server):
        assert "init_project" in {p.name for p in await server.list_prompts()}


class TestModelsResource:
    async def test_returns_parseable_json(self, server):
        blocks = list(await server.read_resource("models://active"))
        assert json.loads(blocks[0].content)

    async def test_names_current_endpoints(self, server):
        payload = json.loads(list(await server.read_resource("models://active"))[0].content)
        assert "anthropic/claude-opus-5" in [m["id"] for m in payload["current"]]

    async def test_maps_superseded_endpoints(self, server):
        payload = json.loads(list(await server.read_resource("models://active"))[0].content)
        assert payload["superseded"]["anthropic/claude-opus-4.8"] == "anthropic/claude-opus-5"

    async def test_states_when_it_was_retrieved(self, server):
        payload = json.loads(list(await server.read_resource("models://active"))[0].content)
        assert payload["fetched_at"].startswith("2026-09-19")

    async def test_excludes_models_failing_the_capability_gate(self, server):
        payload = json.loads(list(await server.read_resource("models://active"))[0].content)
        assert "google/lyria-3-clip-preview" not in [m["id"] for m in payload["current"]]


class TestServerFraming:
    """The server's own words reach the host too: as MCP instructions at connect
    time, as resource descriptions, and as the init_project prompt. The same
    rule as the session block applies (see test_framing)."""

    @staticmethod
    def _assert_neutral(text: str) -> None:
        import re

        from .test_framing import AMBIGUOUS_PATTERNS, OVERRIDE_PATTERNS

        for pattern in OVERRIDE_PATTERNS + AMBIGUOUS_PATTERNS:
            assert not re.search(pattern, text, re.IGNORECASE), pattern

    async def test_instructions_claim_no_authority(self, server):
        self._assert_neutral(server.instructions or "")

    async def test_resource_description_claims_no_authority(self, server):
        resource = next(r for r in await server.list_resources() if str(r.uri) == "models://active")
        self._assert_neutral(resource.description or "")

    async def test_resource_description_is_scoped_to_callable_endpoints(self, server):
        resource = next(r for r in await server.list_resources() if str(r.uri) == "models://active")
        assert "api" in (resource.description or "").lower()

    async def test_init_prompt_claims_no_authority(self, server):
        result = await server.get_prompt("init_project")
        self._assert_neutral("\n".join(getattr(m.content, "text", "") for m in result.messages))

    async def test_instructions_reach_beyond_model_limits(self, server):
        assert "parser" in (server.instructions or "").lower()


class TestInitProjectPrompt:
    async def test_carries_the_active_catalog(self, server):
        result = await server.get_prompt("init_project")
        body = "\n".join(getattr(m.content, "text", "") for m in result.messages)
        assert "anthropic/claude-opus-5" in body

    async def test_tells_the_agent_to_call_check_what_exists(self, server):
        result = await server.get_prompt("init_project")
        body = "\n".join(getattr(m.content, "text", "") for m in result.messages)
        assert "check_what_exists" in body


class TestVerifyArchitectureTool:
    @pytest.fixture
    def wired(self, monkeypatch, raw_models):
        """A server whose LLM and retrieval are substituted, nothing else."""
        from sota_anchor import server as module

        async def fake_fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        monkeypatch.setattr(module, "fetch_catalog", fake_fetch)

        state: dict[str, object] = {"evidence": evidence_set("PlanSightRAG"), "verdict": OBSOLETE}

        async def fake_gather(query, **kwargs):
            state["queried"] = query
            return state["evidence"]

        def fake_llm():
            return FakeLLM(VALID, state["verdict"])

        monkeypatch.setattr(module, "gather_evidence", fake_gather)
        monkeypatch.setattr(module, "build_llm", lambda: fake_llm())
        return module.build_server(), state

    async def test_returns_the_paradigm_update_for_an_obsolete_pitch(self, wired):
        server, _ = wired
        result = await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert "[SOTA ARBITER PARADIGM UPDATE]" in text_of(result)

    async def test_confirms_a_pitch_whose_limitation_still_holds(self, wired):
        server, state = wired
        state["verdict"] = STILL_VALID
        result = await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert "[SOTA ARBITER PARADIGM UPDATE]" not in text_of(result)

    async def test_retrieves_using_the_inverted_query(self, wired):
        server, state = wired
        await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert VALID["domain_query"] in state["queried"]
        assert VALID["capability_query"] in state["queried"]

    async def test_no_evidence_never_yields_an_obsolescence_claim(self, wired):
        server, state = wired
        state["evidence"] = EvidenceSet()
        result = await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert "[SOTA ARBITER PARADIGM UPDATE]" not in text_of(result)


    async def test_an_empty_pitch_is_reported_not_raised(self, wired):
        server, _ = wired
        result = await server.call_tool("check_what_exists", {"pitch": "   "})
        assert text_of(result).strip()


class TestTwoPhaseVerifyArchitecture:
    """The MCP tool with no API key: it returns protocol, not a verdict.

    It stays self-describing on purpose. A caller that never loaded the skill
    still gets told what to do next, rather than a bare evidence dump.
    """

    @pytest.fixture
    def keyless(self, monkeypatch, raw_models):
        from sota_anchor import server as module

        async def fake_fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        state: dict[str, object] = {"evidence": evidence_set("PlanSightRAG")}

        async def fake_gather(query, **kwargs):
            state["queried"] = query
            return state["evidence"]

        def no_key():
            from sota_anchor.arbiter import LLMUnavailable

            raise LLMUnavailable("no API key found")

        monkeypatch.setattr(module, "fetch_catalog", fake_fetch)
        monkeypatch.setattr(module, "gather_evidence", fake_gather)
        monkeypatch.setattr(module, "build_llm", no_key)
        return module.build_server(), state

    async def test_tool_accepts_an_optional_verification_query(self, keyless):
        server, _ = keyless
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        properties = (tool.input_schema or {}).get("properties", {})
        assert "domain_query" in properties
        assert "domain_query" not in (tool.input_schema or {}).get("required", [])

    async def test_without_a_query_it_returns_the_inversion_prompt(self, keyless):
        server, _ = keyless
        result = await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert "ASSUMPTION INVERSION" in text_of(result)

    async def test_without_a_query_it_does_not_retrieve(self, keyless):
        server, state = keyless
        await server.call_tool("check_what_exists", {"pitch": "OCR snapping pipeline"})
        assert "queried" not in state

    async def test_with_a_query_it_returns_the_judging_protocol(self, keyless):
        server, state = keyless
        result = await server.call_tool(
            "check_what_exists",
            {"pitch": "OCR snapping pipeline", "domain_query": "vector polygon extraction"},
        )
        body = text_of(result)
        assert "PARADIGM SHIFT" in body
        assert "PlanSightRAG" in body
        assert "vector polygon extraction" in state["queried"]

    async def test_with_a_query_but_no_evidence_it_refuses_to_invite_a_verdict(self, keyless):
        server, state = keyless
        state["evidence"] = EvidenceSet()
        result = await server.call_tool(
            "check_what_exists",
            {"pitch": "OCR snapping pipeline", "domain_query": "some query"},
        )
        body = text_of(result)
        assert "NO VERDICT POSSIBLE" in body
        assert "PARADIGM SHIFT" not in body

    async def test_no_api_key_is_never_mentioned_as_an_obstacle(self, keyless):
        # Needing no key is the point; the tool must not report its absence as a fault.
        server, _ = keyless
        result = await server.call_tool("check_what_exists", {"pitch": "a pitch"})
        assert "API key" not in text_of(result)

    async def test_an_empty_pitch_is_reported_not_raised(self, keyless):
        server, _ = keyless
        result = await server.call_tool("check_what_exists", {"pitch": "   "})
        assert text_of(result).strip()


class TestWindowParameter:
    """A live run hit this: the host wanted to widen the window after a thin pass
    and found the MCP tool hardcoded 12 months, with the CLI its only way out.
    """

    @pytest.fixture
    def keyless(self, monkeypatch, raw_models):
        from sota_anchor import server as module

        async def fake_fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        state: dict[str, object] = {}

        async def fake_gather(query, **kwargs):
            state["kwargs"] = kwargs
            return evidence_set("PlanSightRAG")

        def no_key():
            from sota_anchor.arbiter import LLMUnavailable

            raise LLMUnavailable("no API key found")

        monkeypatch.setattr(module, "fetch_catalog", fake_fetch)
        monkeypatch.setattr(module, "gather_evidence", fake_gather)
        monkeypatch.setattr(module, "build_llm", no_key)
        return module.build_server(), state

    async def test_tool_exposes_a_months_parameter(self, keyless):
        server, _ = keyless
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        assert "months" in (tool.input_schema or {}).get("properties", {})

    async def test_months_is_optional(self, keyless):
        server, _ = keyless
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        assert "months" not in (tool.input_schema or {}).get("required", [])

    async def test_months_reaches_retrieval(self, keyless):
        server, state = keyless
        await server.call_tool(
            "check_what_exists",
            {"pitch": "a pitch", "domain_query": "a query", "months": 24},
        )
        assert state["kwargs"]["months"] == 24

    async def test_default_window_is_used_when_omitted(self, keyless):
        from sota_anchor.retriever import DEFAULT_WINDOW_MONTHS

        server, state = keyless
        await server.call_tool(
            "check_what_exists", {"pitch": "a pitch", "domain_query": "a query"}
        )
        assert state["kwargs"]["months"] == DEFAULT_WINDOW_MONTHS

    async def test_the_schema_bounds_months(self, keyless):
        server, _ = keyless
        tool = next(t for t in await server.list_tools() if t.name == "check_what_exists")
        months = tool.input_schema["properties"]["months"]
        assert months["minimum"] == 1
        assert months["maximum"] == 120

    @pytest.mark.parametrize("months", [0, -3, 121, 100000])
    async def test_an_out_of_range_window_is_refused_before_retrieval(self, keyless, months):
        from mcp.server.mcpserver.exceptions import ToolError

        server, state = keyless
        with pytest.raises(ToolError, match="months"):
            await server.call_tool(
                "check_what_exists",
                {"pitch": "a pitch", "domain_query": "a query", "months": months},
            )
        assert "kwargs" not in state


class TestCatalogFailureIsNotFatal:
    """The registry snapshot only grounds the judge. Whatever goes wrong while
    loading it, the check still searches and answers."""

    async def test_the_check_answers_when_the_catalog_fails_unexpectedly(self, monkeypatch):
        from sota_anchor import server as module

        async def broken_fetch(**kwargs):
            raise AttributeError("'list' object has no attribute 'get'")

        async def fake_gather(query, **kwargs):
            return evidence_set("PlanSightRAG")

        def no_key():
            from sota_anchor.arbiter import LLMUnavailable

            raise LLMUnavailable("no API key found")

        monkeypatch.setattr(module, "fetch_catalog", broken_fetch)
        monkeypatch.setattr(module, "gather_evidence", fake_gather)
        monkeypatch.setattr(module, "build_llm", no_key)
        result = await module.build_server().call_tool(
            "check_what_exists", {"pitch": "a pitch", "domain_query": "a query"}
        )
        assert "PlanSightRAG" in text_of(result)
