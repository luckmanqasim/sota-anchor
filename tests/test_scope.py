"""The check covers engineering inertia, not only model capability.

A live session was asked to "convert some nwds to glbs, write a custom parer
for it since i cant use oda to read the files". It declined to run the check,
reasoning that verify_architecture only applies to AI model capabilities. But
the assumption under that plan -- no open-source reader exists for the format
without the vendor's SDK -- is exactly the kind this tool exists to test: an
independent reader for it had been published on GitHub two weeks earlier.

So every surface that decides whether the check runs, and how the assumption
is extracted and judged, has to name that class of workaround, not just
workarounds for what a model cannot do.
"""

from __future__ import annotations

from pathlib import Path

from sota_anchor.arbiter import judge
from sota_anchor.protocol import INVERSION_TEMPLATE, JUDGMENT_TEMPLATE
from sota_anchor.server import VERIFY_DESCRIPTION

from .test_arbiter import INVERSION, STILL_VALID, evidence_set
from .test_inversion import VALID, FakeLLM

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "skills" / "sota-architect" / "SKILL.md"


def flat(text: str) -> str:
    """Collapse wrapping and case, so assertions test wording, not layout."""
    return " ".join(text.split()).lower()


def skill_description() -> str:
    front = SKILL.read_text(encoding="utf-8").split("---", 2)[1]
    return flat(front.split("description:", 1)[1])


def skill_body() -> str:
    return flat(SKILL.read_text(encoding="utf-8").split("---", 2)[2])


class TestSkillTrigger:
    """The description is what the host matches a request against."""

    def test_names_custom_parsers_and_converters(self):
        description = skill_description()
        assert "parser" in description
        assert "converter" in description

    def test_names_vendor_tooling_as_a_premise_to_check(self):
        assert "vendor" in skill_description()

    def test_names_rewrites_from_scratch(self):
        assert "from scratch" in skill_description()

    def test_still_covers_model_limitations(self):
        assert "model" in skill_description()


class TestAdvicePath:
    """The check has to run before *advice*, not only before building.

    Measured headless with the plugin loaded, three trials each. "i need to read
    some nwds, write a custom parer for it" checked first every time. "i need to
    convert some nwds to glbs, write a custom parer for it" never did: the model
    advised against the parser from memory -- "Autodesk has never documented
    NWD... nothing open-source reads it" -- which is the stale claim itself. It
    was not building anything, so a trigger worded "use before building around"
    did not apply. In the live session that first answer then anchored the next
    two turns. Naming the advice path took the convert prompt from 0/3 to 3/3,
    with the check as the first action, while two ordinary prompts stayed 0/2.
    """

    def test_the_skill_runs_before_advising_against_building(self):
        assert "advise against" in skill_description()

    def test_the_skill_names_the_claims_training_data_gets_wrong(self):
        description = skill_description()
        assert "no open-source tool exists" in description
        assert "only the vendor" in description

    def test_the_session_block_names_the_advice_path(self, raw_models):
        from sota_anchor.catalog import build_catalog
        from sota_anchor.seed import BOOTSTRAP_BLOCK, render_seed

        from .conftest import NOW

        for block in (render_seed(build_catalog(raw_models, now=NOW)), BOOTSTRAP_BLOCK):
            assert "advising that no open-source tool exists" in flat(block)

    def test_the_mcp_instructions_name_the_advice_path(self):
        from sota_anchor.server import build_server

        assert "advising that no tool exists" in flat(build_server().instructions or "")

    def test_the_tool_description_names_the_advice_path(self):
        assert "no tool or library exists" in flat(VERIFY_DESCRIPTION)


class TestSkillBody:
    def test_inverts_to_what_must_be_unavailable(self):
        assert "unavailable" in skill_body()

    def test_keeps_stated_constraints_as_given(self):
        # "I can't use ODA" is the user's constraint, not a claim to overturn.
        assert "constraint" in skill_body()

    def test_works_an_example_that_is_not_about_a_model(self):
        body = skill_body()
        assert "no open-source reader" in body

    def test_asks_for_the_most_specific_term_first(self):
        assert "most specific term" in skill_body()


class TestToolDescription:
    def test_names_custom_parsers(self):
        assert "parser" in flat(VERIFY_DESCRIPTION)

    def test_says_package_registries_are_searched(self):
        assert "package registries" in flat(VERIFY_DESCRIPTION)

    def test_asks_for_the_most_specific_term_first(self):
        assert "most specific term" in flat(VERIFY_DESCRIPTION)


class TestInversionTemplate:
    def test_names_limitations_beyond_models(self):
        template = flat(INVERSION_TEMPLATE)
        for kind in ("library", "format", "vendor", "api"):
            assert kind in template, kind

    def test_keeps_stated_constraints_as_given(self):
        assert "constraint" in flat(INVERSION_TEMPLATE)

    def test_capability_query_may_name_an_existing_implementation(self):
        assert "existing implementation" in flat(INVERSION_TEMPLATE)

    def test_asks_for_the_most_specific_term_first(self):
        assert "most specific term" in flat(INVERSION_TEMPLATE)


class TestJudgmentStandard:
    """Existence and performance claims are overturned by different evidence.

    "No reader exists for this format" is refuted by a published reader. "Models
    cannot do this accurately" is refuted only by results. One threshold for both
    either ignores working code or accepts unbenchmarked claims.
    """

    def test_distinguishes_existence_from_performance(self):
        template = flat(JUDGMENT_TEMPLATE)
        assert "existence" in template
        assert "performance" in template

    def test_a_published_implementation_documents_existence(self):
        assert "repository or package" in flat(JUDGMENT_TEMPLATE)

    def test_existence_is_not_mistaken_for_maturity(self):
        assert "maturity" in flat(JUDGMENT_TEMPLATE)

    def test_performance_claims_still_need_benchmarks(self):
        assert "benchmark" in flat(JUDGMENT_TEMPLATE)


class TestKeyedPrompts:
    """The keyed, headless path must extract and judge the same way."""

    async def test_inversion_prompt_names_limitations_beyond_models(self):
        from sota_anchor.arbiter import invert

        llm = FakeLLM(VALID)
        await invert("Write a custom parser for format X.", llm=llm)
        prompt = flat(llm.prompts[0])
        assert "library" in prompt
        assert "constraint" in prompt
        assert "most specific term" in prompt

    async def test_judge_prompt_distinguishes_existence_from_performance(self):
        llm = FakeLLM(STILL_VALID)
        await judge(INVERSION, evidence_set("PlanSightRAG"), llm=llm)
        prompt = flat(llm.prompts[0])
        assert "existence" in prompt
        assert "performance" in prompt
