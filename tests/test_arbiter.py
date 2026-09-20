"""Stage 2 and 3: epistemic conflict detection and the paradigm-update output."""

from __future__ import annotations

import datetime as dt

import pytest

from sota_anchor.arbiter import (
    ArbiterError,
    Inversion,
    Report,
    Verdict,
    judge,
    render_paradigm_update,
    verify_architecture,
)
from sota_anchor.retriever import Evidence, EvidenceSet

from .conftest import NOW
from .test_inversion import VALID, FakeLLM

INVERSION = Inversion(**VALID)

OBSOLETE = {
    "is_obsolete": True,
    "assertion": "Do NOT build OCR bounding boxes and geometric snapping heuristics.",
    "reason": (
        "Visual-first multimodal RAG systems now return semantic polygon annotations "
        "from civil standard plans directly."
    ),
    "logical_linkage": (
        "Because visual-first multimodal RAG returns polygons directly, building an OCR "
        "snapping pipeline represents redundant technical debt."
    ),
}

STILL_VALID = {
    "is_obsolete": False,
    "rationale": "No system yet returns dimensioned vector geometry from raster scans.",
}


def evidence_set(*titles: str) -> EvidenceSet:
    return EvidenceSet(
        items=[
            Evidence(
                source="arxiv",
                title=title,
                url=f"http://arxiv.org/abs/2608.0000{index}v1",
                published=dt.date(2026, 8, 26),
                snippet="Direct polygon extraction from technical drawings.",
            )
            for index, title in enumerate(titles, start=1)
        ],
        window_months=12,
    )


class TestJudgePrompt:
    async def test_states_the_proposed_assumption(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        assert INVERSION.implicit_limitation in llm.prompts[0]

    async def test_states_the_proposed_workaround(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        assert INVERSION.proposed_workaround in llm.prompts[0]

    async def test_includes_the_retrieved_evidence(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        assert "PlanSightRAG" in llm.prompts[0]

    async def test_states_the_recency_window(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        assert "12" in llm.prompts[0]

    async def test_asks_for_assertion_reason_and_linkage(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        prompt = llm.prompts[0]
        assert "assertion" in prompt
        assert "reason" in prompt
        assert "logical_linkage" in prompt

    async def test_offers_both_verdicts_so_the_judge_can_say_no(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        assert "is_obsolete" in llm.prompts[0]
        assert "rationale" in llm.prompts[0]


class TestVerdictParsing:
    async def test_parses_an_obsolete_verdict(self):
        verdict = await judge(INVERSION, evidence_set("X"), llm=FakeLLM(OBSOLETE))
        assert verdict.is_obsolete is True
        assert verdict.assertion == OBSOLETE["assertion"]

    async def test_parses_a_still_valid_verdict(self):
        verdict = await judge(INVERSION, evidence_set("X"), llm=FakeLLM(STILL_VALID))
        assert verdict.is_obsolete is False
        assert verdict.rationale == STILL_VALID["rationale"]

    async def test_obsolete_verdict_missing_its_reason_is_rejected(self):
        broken = {k: v for k, v in OBSOLETE.items() if k != "reason"}
        with pytest.raises(ArbiterError):
            await judge(INVERSION, evidence_set("X"), llm=FakeLLM(broken, broken, broken))

    async def test_obsolete_verdict_with_a_blank_reason_is_rejected(self):
        blank = {**OBSOLETE, "reason": ""}
        with pytest.raises(ArbiterError):
            await judge(INVERSION, evidence_set("X"), llm=FakeLLM(blank, blank, blank))

    async def test_string_true_is_accepted_as_obsolete(self):
        verdict = await judge(
            INVERSION, evidence_set("X"), llm=FakeLLM({**OBSOLETE, "is_obsolete": "true"})
        )
        assert verdict.is_obsolete is True


class TestEmptyEvidenceGuard:
    """The worst failure this tool can have is telling someone not to build
    something on the basis of no evidence. The empty path must be unable to
    reach an obsolescence verdict, regardless of what a model would have said.
    """

    async def test_judge_is_not_called_when_no_evidence_was_retrieved(self):
        llm = FakeLLM(OBSOLETE)
        await judge(INVERSION, EvidenceSet(), llm=llm)
        assert llm.prompts == []

    async def test_verdict_is_not_obsolete_when_no_evidence_was_retrieved(self):
        verdict = await judge(INVERSION, EvidenceSet(), llm=FakeLLM(OBSOLETE))
        assert verdict.is_obsolete is False

    async def test_rationale_says_evidence_was_missing_not_that_nothing_changed(self):
        verdict = await judge(INVERSION, EvidenceSet(), llm=FakeLLM(OBSOLETE))
        assert "no recent evidence" in verdict.rationale.lower()

    async def test_retrieval_errors_are_surfaced_in_the_rationale(self):
        empty = EvidenceSet(errors=["arxiv: HTTPStatusError: 429"])
        verdict = await judge(INVERSION, empty, llm=FakeLLM(OBSOLETE))
        assert "arxiv" in verdict.rationale


class TestParadigmUpdateRendering:
    def test_uses_the_specified_block_header(self):
        rendered = render_paradigm_update(Verdict(**OBSOLETE))
        assert rendered.startswith("[SOTA ARBITER PARADIGM UPDATE]")

    def test_labels_assertion_reason_and_linkage(self):
        rendered = render_paradigm_update(Verdict(**OBSOLETE))
        assert "- Assertion (A): " in rendered
        assert "- Reason (R): " in rendered
        assert "- Linkage: " in rendered

    def test_carries_each_field_value(self):
        rendered = render_paradigm_update(Verdict(**OBSOLETE))
        assert OBSOLETE["assertion"] in rendered
        assert OBSOLETE["reason"] in rendered
        assert OBSOLETE["logical_linkage"] in rendered

    def test_refuses_to_render_a_non_obsolete_verdict(self):
        with pytest.raises(ValueError):
            render_paradigm_update(Verdict(**STILL_VALID))

    def test_output_is_ascii_only(self):
        wide = {**OBSOLETE, "reason": "Café — naïve heuristics are obsolete."}
        assert render_paradigm_update(Verdict(**wide)).isascii()


class TestVerifyArchitecture:
    async def _run(self, llm, evidence: EvidenceSet, **kwargs) -> Report:
        async def gather(query, **_):
            self.queried = query
            return evidence

        return await verify_architecture(
            "Build an OCR snapping pipeline for MEP penetrations.",
            llm=llm,
            gather=gather,
            now=NOW,
            **kwargs,
        )

    async def test_retrieves_using_the_inverted_verification_query(self):
        await self._run(FakeLLM(VALID, STILL_VALID), evidence_set("X"))
        assert self.queried == VALID["verification_query"]

    async def test_runs_both_stages_in_order(self):
        llm = FakeLLM(VALID, STILL_VALID)
        await self._run(llm, evidence_set("X"))
        assert "must be hard" in llm.prompts[0].lower() or "impossible" in llm.prompts[0].lower()
        assert "Arbiter" in llm.prompts[1]

    async def test_reports_an_obsolete_proposal(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), evidence_set("PlanSightRAG"))
        assert report.verdict.is_obsolete is True

    async def test_report_renders_the_paradigm_update_when_obsolete(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), evidence_set("PlanSightRAG"))
        assert "[SOTA ARBITER PARADIGM UPDATE]" in report.render()

    async def test_report_confirms_when_the_limitation_still_holds(self):
        report = await self._run(FakeLLM(VALID, STILL_VALID), evidence_set("X"))
        rendered = report.render()
        assert "[SOTA ARBITER PARADIGM UPDATE]" not in rendered
        assert STILL_VALID["rationale"] in rendered

    async def test_report_keeps_the_evidence_for_inspection(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), evidence_set("PlanSightRAG"))
        assert any("PlanSightRAG" in item.title for item in report.evidence.items)

    async def test_report_names_the_domain_the_inverter_identified(self):
        report = await self._run(FakeLLM(VALID, STILL_VALID), evidence_set("X"))
        assert report.inversion.domain == VALID["domain"]

    async def test_report_cites_evidence_urls(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), evidence_set("PlanSightRAG"))
        assert "arxiv.org/abs" in report.render()

    async def test_no_evidence_yields_a_not_obsolete_report(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), EvidenceSet())
        assert report.verdict.is_obsolete is False

    async def test_retrieval_errors_appear_in_the_report(self):
        broken = EvidenceSet(errors=["arxiv: HTTPStatusError: 429"])
        report = await self._run(FakeLLM(VALID, OBSOLETE), broken)
        assert "429" in report.render()

    async def test_report_render_is_ascii_only(self):
        report = await self._run(FakeLLM(VALID, OBSOLETE), evidence_set("Café — study"))
        assert report.render().isascii()

    async def test_window_months_is_passed_through_to_retrieval(self):
        captured: dict[str, object] = {}

        async def gather(query, **kwargs):
            captured.update(kwargs)
            return evidence_set("X")

        await verify_architecture(
            "a proposal", llm=FakeLLM(VALID, STILL_VALID), gather=gather, now=NOW, months=6
        )
        assert captured["months"] == 6
