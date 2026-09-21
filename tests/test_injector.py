"""The injector edits files a developer owns. It must never touch their content."""

from __future__ import annotations

import pytest

from sota_anchor.injector import (
    BLOCK_END,
    BLOCK_START,
    inject,
    resolve_targets,
    sync_targets,
)


@pytest.fixture
def block() -> str:
    return "## Active model endpoints\n\nUse `acme/model-2`."


class TestInject:
    def test_creates_a_missing_file(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        inject(path, block)
        assert BLOCK_START in path.read_text(encoding="utf-8")

    def test_wraps_the_body_in_both_markers(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        inject(path, block)
        text = path.read_text(encoding="utf-8")
        assert text.index(BLOCK_START) < text.index(block) < text.index(BLOCK_END)

    def test_appends_without_disturbing_existing_content(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_text("# My project\n\nHand-written guidance.\n", encoding="utf-8")
        inject(path, block)
        text = path.read_text(encoding="utf-8")
        assert "# My project" in text
        assert "Hand-written guidance." in text

    def test_replaces_an_existing_block_in_place(self, tmp_path):
        path = tmp_path / "CLAUDE.md"
        inject(path, "first body")
        inject(path, "second body")
        text = path.read_text(encoding="utf-8")
        assert "second body" in text
        assert "first body" not in text

    def test_does_not_accumulate_blocks(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        for _ in range(3):
            inject(path, block)
        assert path.read_text(encoding="utf-8").count(BLOCK_START) == 1

    def test_is_byte_identical_when_rerun_with_the_same_body(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_text("# Project\n", encoding="utf-8")
        inject(path, block)
        once = path.read_bytes()
        inject(path, block)
        assert path.read_bytes() == once

    def test_preserves_content_on_both_sides_of_the_block(self, tmp_path):
        path = tmp_path / "CLAUDE.md"
        path.write_text(
            f"before\n{BLOCK_START}\nold\n{BLOCK_END}\nafter\n", encoding="utf-8"
        )
        inject(path, "new")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("before")
        assert text.rstrip().endswith("after")
        assert "old" not in text

    def test_reports_whether_it_changed_anything(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        assert inject(path, block).changed is True
        assert inject(path, block).changed is False

    def test_preserves_crlf_line_endings(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_bytes(b"# Project\r\n\r\nGuidance.\r\n")
        inject(path, block)
        raw = path.read_bytes()
        assert b"\r\n" in raw
        assert b"\n" not in raw.replace(b"\r\n", b"")

    def test_preserves_lf_line_endings(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_bytes(b"# Project\n\nGuidance.\n")
        inject(path, block)
        assert b"\r\n" not in path.read_bytes()

    def test_reads_and_writes_utf8_regardless_of_platform_default(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_text("# Projet café — naive\n", encoding="utf-8")
        inject(path, block)
        assert "café" in path.read_text(encoding="utf-8")

    def test_separates_the_block_from_preceding_prose(self, tmp_path, block):
        path = tmp_path / "AGENTS.md"
        path.write_text("Existing line.", encoding="utf-8")
        inject(path, block)
        assert "Existing line.\n\n" in path.read_text(encoding="utf-8")

    def test_an_unterminated_marker_is_not_silently_mangled(self, tmp_path, block):
        path = tmp_path / "CLAUDE.md"
        path.write_text(f"keep me\n{BLOCK_START}\ntruncated", encoding="utf-8")
        with pytest.raises(ValueError, match="unterminated"):
            inject(path, block)
        assert "keep me" in path.read_text(encoding="utf-8")


class TestCursorMdc:
    def test_adds_frontmatter_when_creating_an_mdc_file(self, tmp_path, block):
        path = tmp_path / ".cursor" / "rules" / "sota.mdc"
        inject(path, block, frontmatter={"alwaysApply": True})
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "alwaysApply: true" in text

    def test_does_not_duplicate_frontmatter_on_update(self, tmp_path, block):
        path = tmp_path / ".cursor" / "rules" / "sota.mdc"
        inject(path, block, frontmatter={"alwaysApply": True})
        inject(path, "changed", frontmatter={"alwaysApply": True})
        assert path.read_text(encoding="utf-8").count("alwaysApply") == 1

    def test_preserves_hand_edited_frontmatter(self, tmp_path, block):
        path = tmp_path / ".cursor" / "rules" / "sota.mdc"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\ndescription: mine\nglobs: ['**/*.py']\n---\n\nbody\n", encoding="utf-8"
        )
        inject(path, block, frontmatter={"alwaysApply": True})
        text = path.read_text(encoding="utf-8")
        assert "description: mine" in text
        assert "globs: ['**/*.py']" in text


class TestResolveTargets:
    def test_claude_target_maps_to_claude_md(self, tmp_path):
        paths = [t.path.name for t in resolve_targets("claude", root=tmp_path)]
        assert paths == ["CLAUDE.md"]

    def test_agents_target_maps_to_agents_md(self, tmp_path):
        paths = [t.path.name for t in resolve_targets("agents", root=tmp_path)]
        assert paths == ["AGENTS.md"]

    def test_cursor_target_covers_both_cursor_conventions(self, tmp_path):
        names = {t.path.name for t in resolve_targets("cursor", root=tmp_path)}
        assert names == {".cursorrules", "sota.mdc"}

    def test_only_the_mdc_target_carries_frontmatter(self, tmp_path):
        targets = {t.path.name: t for t in resolve_targets("cursor", root=tmp_path)}
        assert targets["sota.mdc"].frontmatter
        assert not targets[".cursorrules"].frontmatter

    def test_all_covers_every_target(self, tmp_path):
        names = {t.path.name for t in resolve_targets("all", root=tmp_path)}
        assert names == {"CLAUDE.md", "AGENTS.md", ".cursorrules", "sota.mdc"}

    def test_rejects_an_unknown_target(self, tmp_path):
        with pytest.raises(ValueError, match="unknown target"):
            resolve_targets("emacs", root=tmp_path)


class TestSyncTargets:
    def test_writes_every_requested_target(self, tmp_path, block):
        results = sync_targets("all", block, root=tmp_path)
        assert all(r.path.exists() for r in results)
        assert len(results) == 4

    def test_creates_nested_directories(self, tmp_path, block):
        sync_targets("cursor", block, root=tmp_path)
        assert (tmp_path / ".cursor" / "rules" / "sota.mdc").exists()

    def test_second_run_reports_no_changes(self, tmp_path, block):
        sync_targets("all", block, root=tmp_path)
        again = sync_targets("all", block, root=tmp_path)
        assert [r.changed for r in again] == [False, False, False, False]
