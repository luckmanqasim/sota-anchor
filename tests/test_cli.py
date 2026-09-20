"""CLI behaviour, including the exit codes that let `check` gate CI."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from sota_anchor.catalog import CatalogUnavailable, build_catalog
from sota_anchor.retriever import EvidenceSet

from .conftest import NOW
from .test_arbiter import OBSOLETE, STILL_VALID, evidence_set
from .test_inversion import VALID, FakeLLM


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty working directory for commands that write instruction files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def wired(monkeypatch, raw_models):
    from sota_anchor import cli as module

    state: dict[str, object] = {
        "evidence": evidence_set("PlanSightRAG"),
        "verdict": STILL_VALID,
        "catalog_error": None,
    }

    async def fake_fetch(**kwargs):
        if state["catalog_error"]:
            raise state["catalog_error"]
        state["fetch_kwargs"] = kwargs
        return build_catalog(raw_models, now=NOW, **{
            k: v for k, v in kwargs.items() if k == "staleness_months"
        })

    async def fake_gather(query, **kwargs):
        state["queried"] = query
        state["gather_kwargs"] = kwargs
        return state["evidence"]

    monkeypatch.setattr(module, "fetch_catalog", fake_fetch)
    monkeypatch.setattr(module, "gather_evidence", fake_gather)
    monkeypatch.setattr(module, "build_llm", lambda: FakeLLM(VALID, state["verdict"]))
    return state


class TestSync:
    def test_writes_claude_md_by_default(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["sync"])
        assert result.exit_code == 0, result.output
        import pathlib

        assert "SOTA-ANCHOR:START" in pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")

    def test_target_all_writes_every_file(self, runner, wired, project):
        import pathlib

        from sota_anchor.cli import main

        runner.invoke(main, ["sync", "--target", "all"])
        for name in ("CLAUDE.md", "AGENTS.md", ".cursorrules"):
            assert pathlib.Path(name).exists()
        assert pathlib.Path(".cursor/rules/sota.mdc").exists()

    def test_reports_which_files_changed(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["sync"])
        assert "CLAUDE.md" in result.output

    def test_second_sync_reports_no_change(self, runner, wired, project):
        from sota_anchor.cli import main

        runner.invoke(main, ["sync"])
        result = runner.invoke(main, ["sync"])
        assert "unchanged" in result.output.lower()

    def test_staleness_months_is_passed_through(self, runner, wired, project):
        from sota_anchor.cli import main

        runner.invoke(main, ["sync", "--staleness-months", "3"])
        assert wired["fetch_kwargs"]["staleness_months"] == 3

    def test_all_providers_widens_the_block(self, runner, wired, project):
        import pathlib

        from sota_anchor.cli import main

        runner.invoke(main, ["sync", "--all-providers"])
        wide = pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")
        runner.invoke(main, ["sync"])
        narrow = pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")
        assert "slowcorp/steady-1" in wide
        assert "slowcorp/steady-1" not in narrow

    def test_provider_can_be_named_explicitly(self, runner, wired, project):
        import pathlib

        from sota_anchor.cli import main

        runner.invoke(main, ["sync", "--provider", "slowcorp"])
        written = pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")
        assert "slowcorp/steady-1" in written
        assert "anthropic/claude-opus-5" not in written

    def test_unreachable_catalog_exits_with_an_error(self, runner, wired, project):
        from sota_anchor.cli import main

        wired["catalog_error"] = CatalogUnavailable("offline and no cache")
        result = runner.invoke(main, ["sync"])
        assert result.exit_code == 1
        assert "offline" in result.output

    def test_rejects_an_unknown_target(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["sync", "--target", "emacs"])
        assert result.exit_code != 0


class TestCheck:
    def test_exits_zero_when_the_limitation_still_holds(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["check", "Build an OCR snapping pipeline."])
        assert result.exit_code == 0, result.output

    def test_exits_two_when_the_proposal_is_obsolete(self, runner, wired, project):
        from sota_anchor.cli import main

        wired["verdict"] = OBSOLETE
        result = runner.invoke(main, ["check", "Build an OCR snapping pipeline."])
        assert result.exit_code == 2

    def test_prints_the_paradigm_update_when_obsolete(self, runner, wired, project):
        from sota_anchor.cli import main

        wired["verdict"] = OBSOLETE
        result = runner.invoke(main, ["check", "Build an OCR snapping pipeline."])
        assert "[SOTA ARBITER PARADIGM UPDATE]" in result.output

    def test_prints_the_rationale_when_still_valid(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["check", "Build an OCR snapping pipeline."])
        assert STILL_VALID["rationale"] in result.output

    def test_retrieves_using_the_inverted_query(self, runner, wired, project):
        from sota_anchor.cli import main

        runner.invoke(main, ["check", "Build an OCR snapping pipeline."])
        assert wired["queried"] == VALID["verification_query"]

    def test_months_option_narrows_the_window(self, runner, wired, project):
        from sota_anchor.cli import main

        runner.invoke(main, ["check", "a pitch", "--months", "6"])
        assert wired["gather_kwargs"]["months"] == 6

    def test_json_output_is_machine_readable(self, runner, wired, project):
        import json

        from sota_anchor.cli import main

        result = runner.invoke(main, ["check", "a pitch", "--json"])
        assert json.loads(result.output)["verdict"]["is_obsolete"] is False

    def test_no_evidence_exits_zero(self, runner, wired, project):
        from sota_anchor.cli import main

        wired["evidence"] = EvidenceSet()
        wired["verdict"] = OBSOLETE
        result = runner.invoke(main, ["check", "a pitch"])
        assert result.exit_code == 0

    def test_missing_api_key_exits_one_with_guidance(self, runner, monkeypatch, wired, project):
        from sota_anchor import cli as module
        from sota_anchor.arbiter import LLMUnavailable

        def explode():
            raise LLMUnavailable("no API key found: set SOTA_ANCHOR_API_KEY")

        monkeypatch.setattr(module, "build_llm", explode)
        result = runner.invoke(module.main, ["check", "a pitch"])
        assert result.exit_code == 1
        assert "SOTA_ANCHOR_API_KEY" in result.output

    def test_empty_proposal_exits_one(self, runner, wired, project):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["check", "   "])
        assert result.exit_code == 1


class TestServe:
    def test_runs_the_stdio_transport(self, runner, monkeypatch, wired, project):
        from sota_anchor import cli as module

        used: dict[str, object] = {}

        class FakeServer:
            def run(self, transport="stdio", **kwargs):
                used["transport"] = transport

        monkeypatch.setattr(module, "build_server", lambda: FakeServer())
        result = runner.invoke(module.main, ["serve"])
        assert result.exit_code == 0, result.output
        assert used["transport"] == "stdio"


class TestTopLevel:
    def test_reports_its_version(self, runner):
        from sota_anchor.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_lists_every_command(self, runner):
        from sota_anchor.cli import main

        output = runner.invoke(main, ["--help"]).output
        for command in ("sync", "check", "serve"):
            assert command in output
