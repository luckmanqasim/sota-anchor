"""Host-driven verification payloads.

`protocol.py` is what makes the plugin work without an API key: instead of
calling an LLM, it builds the prompts the *host* Claude session answers. That
inverts who is trusted, so these tests are mostly about what the payloads refuse
to do.
"""

from __future__ import annotations

import datetime as dt

import pytest

from sota_anchor.protocol import (
    PHASE_INVERT,
    PHASE_JUDGE,
    PHASE_NO_EVIDENCE,
    build_inversion_request,
    build_judgment_request,
    build_verification_payload,
)
from sota_anchor.retriever import Evidence, EvidenceSet

PITCH = "Extract MEP pipe penetrations from PDFs using OCR bounding boxes and snapping heuristics."
QUERY = "multimodal VLM direct vector polygon extraction technical drawings"


def evidence_set(*titles: str, errors: list[str] | None = None) -> EvidenceSet:
    return EvidenceSet(
        items=[
            Evidence(
                source="arxiv",
                title=title,
                url=f"http://arxiv.org/abs/2608.0000{index}v1",
                published=dt.date(2026, 8, 26),
                snippet="Direct polygon extraction from civil standard plans.",
            )
            for index, title in enumerate(titles, start=1)
        ],
        errors=errors or [],
        window_months=12,
    )


class TestPurity:
    def test_module_pulls_in_no_llm_client(self):
        # The whole point is that this path needs no provider account.
        import inspect

        import sota_anchor.protocol as protocol

        source = inspect.getsource(protocol)
        assert "openai" not in source.lower()
        assert "LLMClient" not in source

    def test_building_payloads_makes_no_network_calls(self, monkeypatch):
        import httpx

        def forbidden(*args, **kwargs):
            raise AssertionError("protocol must not perform I/O")

        monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
        build_inversion_request(PITCH)
        build_judgment_request(PITCH, QUERY, evidence_set("PlanSightRAG"))


class TestInversionRequest:
    def test_is_labelled_as_the_inversion_phase(self):
        assert build_inversion_request(PITCH).phase == PHASE_INVERT

    def test_quotes_the_pitch_verbatim(self):
        assert PITCH in build_inversion_request(PITCH).render()

    def test_asks_what_must_be_impossible_for_the_design_to_be_justified(self):
        rendered = build_inversion_request(PITCH).render().lower()
        assert "impossible" in rendered
        assert "justified" in rendered

    def test_requests_the_four_inversion_fields(self):
        rendered = build_inversion_request(PITCH).render()
        for field in ("domain", "implicit_limitation", "proposed_workaround", "verification_query"):
            assert field in rendered

    def test_tells_the_host_to_call_back_with_the_query(self):
        rendered = build_inversion_request(PITCH).render()
        assert "verification_query" in rendered
        assert "again" in rendered.lower()

    def test_does_not_ask_the_host_to_decide_obsolescence_yet(self):
        # Judging before evidence exists is precisely the failure being prevented.
        rendered = build_inversion_request(PITCH).render().lower()
        assert "paradigm update" not in rendered

    def test_is_identical_across_unrelated_domains(self):
        a = build_inversion_request("PROPOSAL_A").render().replace("PROPOSAL_A", "X")
        b = build_inversion_request("PROPOSAL_B").render().replace("PROPOSAL_B", "X")
        assert a == b

    def test_rejects_an_empty_pitch(self):
        with pytest.raises(ValueError):
            build_inversion_request("   ")

    def test_render_is_ascii_only(self):
        assert build_inversion_request("Café — naïve OCR").render().isascii()


class TestJudgmentRequest:
    def _render(self, *titles: str) -> str:
        return build_judgment_request(PITCH, QUERY, evidence_set(*titles)).render()

    def test_is_labelled_as_the_judging_phase(self):
        request = build_judgment_request(PITCH, QUERY, evidence_set("PlanSightRAG"))
        assert request.phase == PHASE_JUDGE

    def test_includes_the_retrieved_evidence(self):
        assert "PlanSightRAG" in self._render("PlanSightRAG")

    def test_cites_evidence_urls(self):
        assert "arxiv.org/abs" in self._render("PlanSightRAG")

    def test_states_publication_dates(self):
        assert "2026-08-26" in self._render("PlanSightRAG")

    def test_states_the_recency_window(self):
        assert "12" in self._render("PlanSightRAG")

    def test_supplies_the_assertion_reason_template(self):
        rendered = self._render("PlanSightRAG")
        assert "[SOTA ARBITER PARADIGM SHIFT]" in rendered
        assert "Assertion (A)" in rendered
        assert "Reason (R)" in rendered
        assert "Linkage" in rendered

    def test_forbids_judging_from_memory(self):
        rendered = self._render("PlanSightRAG").lower()
        assert "evidence above" in rendered
        assert "training" in rendered or "memory" in rendered

    def test_offers_the_not_obsolete_outcome_too(self):
        # A protocol that only describes how to say YES is a protocol for saying YES.
        assert "still" in self._render("PlanSightRAG").lower()

    def test_carries_the_original_pitch_for_context(self):
        assert PITCH in self._render("PlanSightRAG")

    def test_reports_partial_retrieval_failures(self):
        request = build_judgment_request(
            PITCH, QUERY, evidence_set("PlanSightRAG", errors=["arxiv: throttled (HTTP 406)"])
        )
        assert "throttled" in request.render()

    def test_render_is_ascii_only(self):
        wide = EvidenceSet(
            items=[
                Evidence(
                    source="arxiv",
                    title="Café — study",
                    url="http://arxiv.org/abs/1",
                    published=dt.date(2026, 8, 1),
                    snippet="Naïve baseline — obsolete.",
                )
            ],
            window_months=12,
        )
        assert build_judgment_request(PITCH, QUERY, wide).render().isascii()


class TestEmptyEvidenceRefusal:
    """The guard that matters most once the judge is the host model.

    Previously this codebase owned the judge call and could decline to make it.
    Now the "judge" is a prompt handed to a model whose priors are the problem.
    Handing over an Assertion-Reason template with an empty evidence block invites
    it to fill the template from memory, which is the exact failure being fixed.
    """

    def _empty(self, errors: list[str] | None = None):
        return build_judgment_request(PITCH, QUERY, EvidenceSet(errors=errors or []))

    def test_is_labelled_as_the_refusal_phase(self):
        assert self._empty().phase == PHASE_NO_EVIDENCE

    def test_withholds_the_assertion_reason_template_entirely(self):
        assert "[SOTA ARBITER PARADIGM SHIFT]" not in self._empty().render()

    def test_does_not_supply_an_assertion_field_to_fill(self):
        rendered = self._empty().render()
        assert "Assertion (A)" not in rendered

    def test_says_plainly_that_nothing_was_retrieved(self):
        assert "no recent evidence" in self._empty().render().lower()

    def test_instructs_the_host_not_to_conclude_obsolescence(self):
        rendered = self._empty().render().lower()
        assert "do not" in rendered

    def test_distinguishes_absent_evidence_from_a_settled_question(self):
        # "Nothing was found" must never be read as "the limitation holds".
        rendered = self._empty().render().lower()
        assert "not " in rendered
        assert "confirm" not in rendered.split("do not")[0]

    def test_surfaces_retrieval_errors_so_silence_is_explainable(self):
        rendered = self._empty(errors=["arxiv: throttled (HTTP 406)"]).render()
        assert "throttled" in rendered

    def test_tells_the_host_to_proceed_with_the_original_plan(self):
        assert "proceed" in self._empty().render().lower()

    def test_render_is_ascii_only(self):
        assert self._empty().render().isascii()


class TestVerificationPayload:
    """The dispatcher the MCP tool and CLI both use."""

    async def test_without_a_query_it_asks_for_the_inversion(self):
        payload = await build_verification_payload(PITCH, verification_query=None)
        assert payload.phase == PHASE_INVERT

    async def test_without_a_query_it_retrieves_nothing(self):
        async def forbidden(*args, **kwargs):
            raise AssertionError("retrieval must wait for the inverted query")

        payload = await build_verification_payload(
            PITCH, verification_query=None, gather=forbidden
        )
        assert payload.phase == PHASE_INVERT

    async def test_with_a_query_it_retrieves_and_asks_for_judgment(self):
        async def gather(query, **kwargs):
            assert query == QUERY
            return evidence_set("PlanSightRAG")

        payload = await build_verification_payload(PITCH, verification_query=QUERY, gather=gather)
        assert payload.phase == PHASE_JUDGE

    async def test_with_a_query_but_no_results_it_refuses(self):
        async def gather(query, **kwargs):
            return EvidenceSet()

        payload = await build_verification_payload(PITCH, verification_query=QUERY, gather=gather)
        assert payload.phase == PHASE_NO_EVIDENCE

    async def test_months_is_passed_through_to_retrieval(self):
        captured: dict[str, object] = {}

        async def gather(query, **kwargs):
            captured.update(kwargs)
            return evidence_set("X")

        await build_verification_payload(
            PITCH, verification_query=QUERY, gather=gather, months=6
        )
        assert captured["months"] == 6

    async def test_a_retrieval_outage_refuses_rather_than_judging(self):
        async def gather(query, **kwargs):
            return EvidenceSet(errors=["arxiv: throttled (HTTP 406)", "github: ConnectError"])

        payload = await build_verification_payload(PITCH, verification_query=QUERY, gather=gather)
        assert payload.phase == PHASE_NO_EVIDENCE
        assert "throttled" in payload.render()
