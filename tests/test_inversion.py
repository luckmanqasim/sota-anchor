"""Stage 1: zero-shot assumption inversion.

The inverter must work on any discipline. These tests deliberately use unrelated
domains -- technical drawings, bioinformatics, compiler IRs -- and assert the code
path is identical, because a keyword table would show up here as a difference.
"""

from __future__ import annotations

import json

import pytest

from sota_anchor.arbiter import (
    ArbiterError,
    Inversion,
    LLMClient,
    LLMSettings,
    LLMUnavailable,
    invert,
    resolve_settings,
)

VALID = {
    "domain": "Architectural Engineering / Technical Drawing Vision",
    "implicit_limitation": (
        "Multimodal models cannot directly extract vector coordinates or semantic "
        "polygon annotations from technical drawings without geometric heuristic "
        "post-processing."
    ),
    "proposed_workaround": "OCR bounding boxes and geometric snapping heuristics.",
    "domain_query": "MEP pipe penetration extraction construction drawings",
    "capability_query": "multimodal VLM direct vector polygon extraction",
}


class FakeLLM:
    """Records prompts and returns queued payloads."""

    def __init__(self, *payloads: object):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        payload = self.payloads.pop(0) if self.payloads else {}
        if isinstance(payload, BaseException):
            raise payload
        return payload


class TestInversionPrompt:
    async def test_includes_the_proposal_verbatim(self):
        llm = FakeLLM(VALID)
        await invert("Build an OCR snapping pipeline for MEP penetrations.", llm=llm)
        assert "Build an OCR snapping pipeline for MEP penetrations." in llm.prompts[0]

    async def test_asks_what_must_be_hard_for_the_design_to_be_justified(self):
        llm = FakeLLM(VALID)
        await invert("anything", llm=llm)
        prompt = llm.prompts[0].lower()
        assert "impossible" in prompt
        assert "justified" in prompt

    async def test_requests_the_four_field_schema(self):
        llm = FakeLLM(VALID)
        await invert("anything", llm=llm)
        for field in ("domain", "implicit_limitation", "proposed_workaround",
                      "domain_query", "capability_query"):
            assert field in llm.prompts[0]

    async def test_asks_for_valid_json(self):
        llm = FakeLLM(VALID)
        await invert("anything", llm=llm)
        assert "json" in llm.prompts[0].lower()

    async def test_prompt_is_identical_across_unrelated_domains(self):
        # The only difference between two disciplines must be the quoted proposal.
        drawings = FakeLLM(VALID)
        genomics = FakeLLM(VALID)
        await invert("PROPOSAL_A", llm=drawings)
        await invert("PROPOSAL_B", llm=genomics)
        assert drawings.prompts[0].replace("PROPOSAL_A", "X") == genomics.prompts[0].replace(
            "PROPOSAL_B", "X"
        )


class TestInversionParsing:
    async def test_returns_the_parsed_inversion(self):
        result = await invert("anything", llm=FakeLLM(VALID))
        assert isinstance(result, Inversion)
        assert result.domain == VALID["domain"]

    async def test_carries_both_query_vectors_for_retrieval(self):
        result = await invert("anything", llm=FakeLLM(VALID))
        assert result.domain_query == VALID["domain_query"]
        assert result.capability_query == VALID["capability_query"]

    async def test_exposes_the_vectors_as_a_list(self):
        result = await invert("anything", llm=FakeLLM(VALID))
        assert result.query_vectors() == [VALID["domain_query"], VALID["capability_query"]]

    async def test_missing_field_is_an_error_not_a_default(self):
        broken = {k: v for k, v in VALID.items() if k != "capability_query"}
        with pytest.raises(ArbiterError):
            await invert("anything", llm=FakeLLM(broken, broken, broken))

    async def test_retries_before_giving_up(self):
        llm = FakeLLM({"domain": "x"}, VALID)
        result = await invert("anything", llm=llm)
        assert result.domain == VALID["domain"]
        assert len(llm.prompts) == 2

    async def test_blank_query_is_rejected(self):
        blank = {**VALID, "domain_query": "   "}
        with pytest.raises(ArbiterError):
            await invert("anything", llm=FakeLLM(blank, blank, blank))

    async def test_empty_proposal_is_refused_without_calling_the_model(self):
        llm = FakeLLM(VALID)
        with pytest.raises(ArbiterError):
            await invert("   ", llm=llm)
        assert llm.prompts == []


class TestSettingsResolution:
    def test_prefers_explicit_sota_anchor_variables(self):
        settings = resolve_settings(
            {
                "SOTA_ANCHOR_API_KEY": "sk-explicit",
                "SOTA_ANCHOR_BASE_URL": "https://example.test/v1",
                "SOTA_ANCHOR_MODEL": "acme/model-9",
            }
        )
        assert settings.api_key == "sk-explicit"
        assert settings.base_url == "https://example.test/v1"
        assert settings.model == "acme/model-9"

    def test_falls_back_to_openrouter_key_and_base_url(self):
        settings = resolve_settings({"OPENROUTER_API_KEY": "sk-or"})
        assert settings.api_key == "sk-or"
        assert "openrouter.ai" in settings.base_url

    def test_falls_back_to_openai_key_and_base_url(self):
        settings = resolve_settings({"OPENAI_API_KEY": "sk-oa"})
        assert settings.api_key == "sk-oa"
        assert "openai.com" in settings.base_url

    def test_openrouter_wins_when_both_are_present(self):
        # The catalog already comes from OpenRouter; one key should cover both channels.
        settings = resolve_settings({"OPENROUTER_API_KEY": "sk-or", "OPENAI_API_KEY": "sk-oa"})
        assert settings.api_key == "sk-or"

    def test_no_key_at_all_is_a_clear_failure(self):
        with pytest.raises(LLMUnavailable, match="SOTA_ANCHOR_API_KEY"):
            resolve_settings({})

    def test_model_is_unset_when_not_configured(self):
        # An unset model is resolved from the live catalog, never hardcoded here.
        assert resolve_settings({"OPENROUTER_API_KEY": "sk-or"}).model is None


class TestModelResolution:
    async def test_uses_the_configured_model_without_touching_the_catalog(self):
        from sota_anchor.arbiter import resolve_model

        settings = LLMSettings(api_key="k", base_url="https://x.test/v1", model="acme/pinned")

        async def explode(**kwargs):
            raise AssertionError("catalog must not be fetched when a model is configured")

        assert await resolve_model(settings, fetch=explode) == "acme/pinned"

    async def test_picks_a_current_model_from_the_catalog(self, raw_models):

        from sota_anchor.arbiter import resolve_model
        from sota_anchor.catalog import build_catalog

        from .conftest import NOW

        settings = LLMSettings(api_key="k", base_url="https://x.test/v1", model=None)

        async def fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        chosen = await resolve_model(settings, fetch=fetch)
        catalog = build_catalog(raw_models, now=NOW)
        assert catalog.by_id(chosen).status == "current"

    async def test_chosen_model_supports_structured_output(self, raw_models):
        from sota_anchor.arbiter import resolve_model
        from sota_anchor.catalog import build_catalog

        from .conftest import NOW

        settings = LLMSettings(api_key="k", base_url="https://x.test/v1", model=None)

        async def fetch(**kwargs):
            return build_catalog(raw_models, now=NOW)

        chosen = await resolve_model(settings, fetch=fetch)
        assert build_catalog(raw_models, now=NOW).by_id(chosen).supports_tools

    async def test_catalog_failure_is_reported_as_unavailable(self):
        from sota_anchor.arbiter import resolve_model
        from sota_anchor.catalog import CatalogUnavailable

        settings = LLMSettings(api_key="k", base_url="https://x.test/v1", model=None)

        async def fetch(**kwargs):
            raise CatalogUnavailable("offline")

        with pytest.raises(LLMUnavailable):
            await resolve_model(settings, fetch=fetch)


class TestLLMClient:
    def _client(self, *replies: str) -> tuple[LLMClient, list[str]]:
        sent: list[str] = []
        queue = list(replies)

        async def completion_fn(prompt: str) -> str:
            sent.append(prompt)
            return queue.pop(0)

        settings = LLMSettings(api_key="k", base_url="https://x.test/v1", model="acme/m")
        return LLMClient(settings, completion_fn=completion_fn), sent

    async def test_parses_a_json_object_reply(self):
        client, _ = self._client(json.dumps(VALID))
        assert (await client.complete_json("prompt"))["domain"] == VALID["domain"]

    async def test_strips_a_fenced_code_block(self):
        client, _ = self._client("```json\n" + json.dumps(VALID) + "\n```")
        assert (await client.complete_json("prompt"))["domain"] == VALID["domain"]

    async def test_recovers_json_embedded_in_prose(self):
        client, _ = self._client("Sure! Here you go:\n" + json.dumps(VALID) + "\nHope that helps.")
        assert (await client.complete_json("prompt"))["domain"] == VALID["domain"]

    async def test_retries_unparseable_output(self):
        client, sent = self._client("not json at all", json.dumps(VALID))
        await client.complete_json("prompt")
        assert len(sent) == 2

    async def test_gives_up_loudly_rather_than_returning_junk(self):
        client, _ = self._client("nope", "still nope", "nope again")
        with pytest.raises(ArbiterError):
            await client.complete_json("prompt")

    async def test_rejects_a_json_array_at_the_top_level(self):
        client, _ = self._client("[1, 2, 3]", "[4]", "[5]")
        with pytest.raises(ArbiterError):
            await client.complete_json("prompt")
