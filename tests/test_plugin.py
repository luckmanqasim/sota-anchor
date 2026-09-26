"""The plugin manifests and hook scripts.

These assert the things that break silently. A hook that points at a missing
script, or omits a timeout on an event whose default is 600 seconds, fails in a
way no unit test of the Python core would ever notice.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLUGIN_MANIFEST = REPO / ".claude-plugin" / "plugin.json"
HOOKS_MANIFEST = REPO / "hooks" / "hooks.json"
MCP_MANIFEST = REPO / ".mcp.json"
#: The server is named for its job: Claude Code shows it as plugin:sota-anchor:check.
MCP_SERVER = "check"
SKILL = REPO / "skills" / "check-what-exists" / "SKILL.md"
COMMANDS = REPO / "commands"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestPluginManifest:
    def test_exists_where_claude_code_looks_for_it(self):
        assert PLUGIN_MANIFEST.is_file()

    def test_is_valid_json(self):
        assert isinstance(load(PLUGIN_MANIFEST), dict)

    def test_declares_a_name_matching_the_package(self):
        assert load(PLUGIN_MANIFEST)["name"] == "sota-anchor"

    def test_declares_a_version(self):
        assert load(PLUGIN_MANIFEST)["version"]

    def test_version_matches_pyproject(self):
        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        version = load(PLUGIN_MANIFEST)["version"]
        assert f'version = "{version}"' in pyproject

    def test_description_says_what_it_does(self):
        assert len(load(PLUGIN_MANIFEST)["description"]) > 40

    def test_description_mentions_needing_no_api_key(self):
        # The headline property of this refactor; it belongs in the listing.
        assert "key" in load(PLUGIN_MANIFEST)["description"].lower()


class TestHooksManifest:
    def test_lives_in_hooks_not_in_the_claude_plugin_dir(self):
        # Verified against seven installed plugins: hooks/hooks.json is the path.
        assert HOOKS_MANIFEST.is_file()
        assert not (REPO / ".claude-plugin" / "hooks.json").exists()

    def test_is_valid_json(self):
        assert "hooks" in load(HOOKS_MANIFEST)

    def test_registers_session_start(self):
        assert "SessionStart" in load(HOOKS_MANIFEST)["hooks"]

    def test_registers_user_prompt_submit(self):
        assert "UserPromptSubmit" in load(HOOKS_MANIFEST)["hooks"]

    def test_session_start_matches_startup_clear_and_compact(self):
        entry = load(HOOKS_MANIFEST)["hooks"]["SessionStart"][0]
        assert entry["matcher"] == "startup|clear|compact"

    def _all_hooks(self) -> list[dict]:
        manifest = load(HOOKS_MANIFEST)["hooks"]
        return [hook for entries in manifest.values() for e in entries for hook in e["hooks"]]

    def test_every_hook_declares_an_explicit_timeout(self):
        # SessionStart defaults to 600s: a hung fetch would stall session startup.
        for hook in self._all_hooks():
            assert "timeout" in hook

    def test_no_timeout_is_long_enough_to_be_noticed(self):
        for hook in self._all_hooks():
            assert hook["timeout"] <= 15

    def test_every_hook_is_a_command_hook(self):
        for hook in self._all_hooks():
            assert hook["type"] == "command"

    def test_commands_go_through_the_plugin_root_variable(self):
        # Absolute or relative paths break as soon as the plugin is installed.
        for hook in self._all_hooks():
            assert "${CLAUDE_PLUGIN_ROOT}" in hook["command"]

    def test_commands_reference_scripts_that_exist(self):
        for hook in self._all_hooks():
            names = [
                token.strip('"').split("/")[-1]
                for token in hook["command"].split()
                if "CLAUDE_PLUGIN_ROOT" in token
            ]
            launcher = names[0]
            assert (REPO / "hooks" / launcher).is_file(), launcher
            script = hook["command"].split()[-1].strip('"')
            assert (REPO / "hooks" / script).is_file(), script

    def test_hook_scripts_avoid_the_sh_extension(self):
        # Claude Code on Windows prepends `bash` to any command containing .sh,
        # which double-invokes through the polyglot launcher.
        for hook in self._all_hooks():
            assert ".sh" not in hook["command"]


class TestMcpManifest:
    def test_ships_the_optional_mcp_server(self):
        assert MCP_MANIFEST.is_file()

    def test_declares_the_server_under_mcp_servers(self):
        assert MCP_SERVER in load(MCP_MANIFEST)["mcpServers"]

    def test_runs_the_stdio_serve_command(self):
        server = load(MCP_MANIFEST)["mcpServers"][MCP_SERVER]
        assert "serve" in server["args"]

    def test_requires_no_api_key_to_start(self):
        server = load(MCP_MANIFEST)["mcpServers"][MCP_SERVER]
        env = server.get("env") or {}
        assert not [key for key in env if "API_KEY" in key and env[key]]


class TestSkill:
    def test_exists_at_the_conventional_path(self):
        assert SKILL.is_file()

    def _frontmatter(self) -> str:
        text = SKILL.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        return text.split("---", 2)[1]

    def test_declares_a_name_and_description(self):
        front = self._frontmatter()
        assert "name:" in front
        assert "description:" in front

    def test_description_says_when_to_use_it(self):
        assert "use" in self._frontmatter().lower()

    def test_body_walks_the_two_phase_protocol(self):
        body = SKILL.read_text(encoding="utf-8")
        assert "domain_query" in body
        assert "capability_query" in body

    def test_body_forbids_judging_from_training_data(self):
        body = SKILL.read_text(encoding="utf-8").lower()
        assert "training" in body

    def test_body_covers_the_no_evidence_outcome(self):
        body = SKILL.read_text(encoding="utf-8").lower()
        assert "empty evidence set" in body
        assert "nothing was checked" in body

    def test_body_names_no_specific_domain(self):
        # The engine is domain-agnostic; the skill must not smuggle in a topic list.
        body = SKILL.read_text(encoding="utf-8").lower()
        for topic in ("autocad", "revit", "bioinformatics"):
            assert topic not in body

    def test_is_ascii_only(self):
        assert SKILL.read_text(encoding="utf-8").isascii()


class TestCommands:
    @pytest.mark.parametrize("name", ["sota-check.md", "sota-sync.md"])
    def test_command_exists(self, name):
        assert (COMMANDS / name).is_file()

    @pytest.mark.parametrize("name", ["sota-check.md", "sota-sync.md"])
    def test_command_declares_a_description(self, name):
        text = (COMMANDS / name).read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "description:" in text.split("---", 2)[1]

    def test_check_declares_an_argument_hint(self):
        front = (COMMANDS / "sota-check.md").read_text(encoding="utf-8").split("---", 2)[1]
        assert "argument-hint:" in front

    def test_check_consumes_the_argument(self):
        body = (COMMANDS / "sota-check.md").read_text(encoding="utf-8")
        assert "$ARGUMENTS" in body

    @pytest.mark.parametrize("name", ["sota-check.md", "sota-sync.md"])
    def test_command_is_ascii_only(self, name):
        assert (COMMANDS / name).read_text(encoding="utf-8").isascii()


class TestHookScripts:
    def test_launcher_is_present(self):
        assert (REPO / "hooks" / "run-hook.cmd").is_file()

    def test_launcher_handles_both_platforms(self):
        text = (REPO / "hooks" / "run-hook.cmd").read_text(encoding="utf-8")
        assert "@echo off" in text
        assert "exec bash" in text

    def test_launcher_degrades_silently_without_bash(self):
        # A seeding hook must never be why a session fails to start.
        text = (REPO / "hooks" / "run-hook.cmd").read_text(encoding="utf-8")
        assert "exit /b 0" in text

    def test_session_start_script_exists(self):
        assert (REPO / "hooks" / "session-start").is_file()

    def test_session_start_emits_the_claude_code_hook_shape(self):
        text = (REPO / "hooks" / "session-start").read_text(encoding="utf-8")
        assert "hookSpecificOutput" in text
        assert "additionalContext" in text

    def test_session_start_always_exits_zero(self):
        text = (REPO / "hooks" / "session-start").read_text(encoding="utf-8")
        assert "exit 0" in text

    def test_session_start_reads_the_prerendered_block(self):
        text = (REPO / "hooks" / "session-start").read_text(encoding="utf-8")
        assert "session-block.md" in text

    def test_session_start_does_not_fetch_on_the_happy_path(self):
        # Sub-second budget: no interpreter, no network while the user waits.
        text = (REPO / "hooks" / "session-start").read_text(encoding="utf-8")
        assert "curl" not in text

    def test_prompt_hook_script_exists(self):
        assert (REPO / "hooks" / "user-prompt-submit").is_file()

    def test_prompt_hook_is_inert_unless_opted_in(self):
        text = (REPO / "hooks" / "user-prompt-submit").read_text(encoding="utf-8")
        assert "SOTA_ANCHOR_PROMPT_HOOK" in text

    def test_prompt_hook_never_blocks_the_prompt(self):
        # Exit 2 on UserPromptSubmit erases what the user typed. Checked against
        # executable lines only: the script's comments discuss exit 2 to explain
        # why it is avoided.
        text = (REPO / "hooks" / "user-prompt-submit").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert '"decision"' not in code
        assert "exit 2" not in code
        assert "exit 0" in code

    @pytest.mark.parametrize("script", ["run-hook.cmd", "session-start", "user-prompt-submit"])
    def test_scripts_are_committed_executable(self, script):
        # hooks.json runs "${CLAUDE_PLUGIN_ROOT}/hooks/run-hook.cmd" directly.
        # Committed as 100644, a clone on Linux or macOS got -rw-r--r-- and
        # every hook failed with "Permission denied" (exit 126): measured in a
        # fresh WSL clone. Windows never noticed, having no execute bit.
        import shutil

        if not (REPO / ".git").exists() or not shutil.which("git"):
            pytest.skip("not a git checkout")
        entry = subprocess.run(
            ["git", "ls-files", "-s", f"hooks/{script}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert entry.startswith("100755"), entry

    @pytest.mark.parametrize("script", ["session-start", "user-prompt-submit"])
    def test_scripts_are_lf_terminated(self, script):
        # CRLF in a shebang script makes bash fail with a cryptic \\r error.
        raw = (REPO / "hooks" / script).read_bytes()
        assert b"\r\n" not in raw

    @pytest.mark.parametrize("script", ["session-start", "user-prompt-submit"])
    def test_scripts_declare_a_shebang(self, script):
        raw = (REPO / "hooks" / script).read_bytes()
        assert raw.startswith(b"#!/usr/bin/env bash")


class TestHookExecution:
    """Run the hooks for real and parse what they emit.

    The text-inspection tests above all passed while the escape function was
    silently broken -- a heredoc had collapsed `\\n` to `\n`, so every newline
    in the injected block became a literal "n". Only executing the hook and
    parsing its output catches that.
    """

    HOOKS = REPO / "hooks"

    @staticmethod
    def _bash() -> str:
        """Resolve a bash that can open a Windows path.

        A bare "bash" resolves to WSL's bash on this platform, which reports
        `No such file or directory` for a C:/... path because it expects
        /mnt/c/... . This mirrors the resolution ladder in run-hook.cmd.
        """
        import shutil

        for candidate in (
            "C:/Program Files/Git/bin/bash.exe",
            "C:/Program Files (x86)/Git/bin/bash.exe",
            "C:/Program Files/Git/usr/bin/bash.exe",
        ):
            if Path(candidate).is_file():
                return candidate
        found = shutil.which("bash")
        if not found:
            pytest.skip("no bash available to execute hook scripts")
        return found

    @staticmethod
    def _posix(path) -> str:
        """bash eats backslashes in an unquoted argument, so hand it forward slashes."""
        return str(path).replace("\\", "/")

    #: Stands in for every tool the refresh ladder can reach. It records how it
    #: was called and touches nothing -- without it, a hook run with no cached
    #: block spawned a real `sota-anchor seed --refresh`, which fetched the live
    #: catalog and overwrote the developer's own cache.
    FAKE_TOOL = (
        "#!/usr/bin/env bash\n"
        "{\n"
        '  printf "tool=%s\\n" "$(basename "$0")"\n'
        '  printf "args=%s\\n" "$*"\n'
        '  printf "cache=%s\\n" "${SOTA_ANCHOR_CACHE_DIR:-}"\n'
        '} >> "${SOTA_FAKE_LOG:?}"\n'
    )

    @pytest.fixture(autouse=True)
    def _fake_refresh_tools(self, tmp_path_factory):
        bin_dir = tmp_path_factory.mktemp("fake-bin")
        for name in ("uv", "sota-anchor"):
            tool = bin_dir / name
            tool.write_text(self.FAKE_TOOL, encoding="utf-8", newline="\n")
            tool.chmod(0o755)
        self.fake_bin = bin_dir
        self.refresh_log = bin_dir / "refresh.log"

    def _refresh_record(self, timeout: float = 10.0) -> dict[str, str]:
        """Wait for the detached refresh to record itself, then parse it."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.refresh_log.is_file():
                text = self.refresh_log.read_text(encoding="utf-8")
                if "cache=" in text:
                    return dict(
                        line.split("=", 1) for line in text.splitlines() if "=" in line
                    )
            time.sleep(0.05)
        raise AssertionError("the hook never launched a refresh")

    def _run(
        self,
        script: str,
        *,
        env: dict[str, str],
        stdin: str = "",
        unset: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess:
        inherited = {k: v for k, v in os.environ.items() if k not in unset}
        merged = {**inherited, **{k: self._posix(v) for k, v in env.items()}}
        merged["PATH"] = str(self.fake_bin) + os.pathsep + os.environ.get("PATH", "")
        merged["SOTA_FAKE_LOG"] = self._posix(self.refresh_log)
        return subprocess.run(
            [self._bash(), self._posix(self.HOOKS / script)],
            capture_output=True,
            text=True,
            input=stdin,
            env=merged,
            timeout=30,
        )

    @pytest.fixture
    def seeded_cache(self, tmp_path):
        from sota_anchor.catalog import build_catalog
        from sota_anchor.seed import SEED_FILENAME, render_seed, write_seed

        from .conftest import NOW

        write_seed(render_seed(build_catalog([], now=NOW)), path=tmp_path / SEED_FILENAME)
        return tmp_path

    def test_session_start_exits_zero_with_no_cache_at_all(self, tmp_path):
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path / "absent"), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        assert result.returncode == 0

    def test_session_start_emits_parseable_json(self, tmp_path):
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path / "absent"), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        payload = json.loads(result.stdout)
        assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_newlines_survive_as_newlines(self, tmp_path):
        # The exact regression: "[SOTA ANCHOR]nnNo model catalog..." instead of
        # real line breaks.
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path / "absent"), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "\n" in context
        assert "]nn" not in context

    def test_the_shipped_bootstrap_survives_intact(self, tmp_path):
        # It carries double quotes and backticks; a broken escape breaks the
        # JSON, and a lossy one changes the text. Compare the whole thing.
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path / "absent"), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        shipped = (self.HOOKS / "bootstrap-block.md").read_text(encoding="utf-8")
        assert '"' in shipped
        assert context == shipped.rstrip("\n")

    def test_the_cached_block_is_what_gets_injected(self, tmp_path):
        from sota_anchor.seed import SEED_FILENAME, write_seed

        block = 'line one\nline "two"\n\tindented\\escaped'
        write_seed(block, path=tmp_path / SEED_FILENAME)
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert context == block

    def test_bootstrap_is_used_only_when_nothing_is_cached(self, tmp_path):
        from sota_anchor.seed import SEED_FILENAME, write_seed

        write_seed("cached block", path=tmp_path / SEED_FILENAME)
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert context == "cached block"
        assert "No model catalog has been fetched" not in context

    def test_falls_back_to_the_sdk_shape_off_claude_code(self, tmp_path):
        from sota_anchor.seed import SEED_FILENAME, write_seed

        write_seed("cached block", path=tmp_path / SEED_FILENAME)
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": ""},
        )
        payload = json.loads(result.stdout)
        assert payload["additionalContext"] == "cached block"
        assert "hookSpecificOutput" not in payload

    def test_carriage_returns_are_stripped_from_the_injected_block(self, tmp_path):
        # A clone with core.autocrlf=true delivered CRLF files, and every line
        # of injected context arrived carrying a literal \r.
        from sota_anchor.seed import SEED_FILENAME

        (tmp_path / SEED_FILENAME).write_bytes(b"line one\r\nline two\r\n")
        result = self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "\r" not in context
        assert context == "line one\nline two"

    def test_refresh_runs_the_plugin_checkout_not_the_callers_project(self, tmp_path):
        # A bare `uv run sota-anchor` resolves against the session's working
        # directory: the user's project. There it found no sota-anchor, so no
        # refresh ever succeeded, and in a project with a pyproject.toml it
        # created a .venv and a uv.lock the user never asked for.
        self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path / "absent"), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        record = self._refresh_record()
        assert record["tool"] == "uv"
        assert record["args"] == (
            f"run --quiet --project {self._posix(REPO)} sota-anchor seed --refresh"
        )

    def test_refresh_writes_where_the_hook_reads(self, tmp_path):
        target = tmp_path / "custom-cache"
        self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(target), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        assert self._refresh_record()["cache"].endswith(f"{tmp_path.name}/custom-cache")

    def test_refresh_is_handed_the_resolved_default_directory(self, tmp_path):
        # Unset override: bash resolves the default from HOME, while Python on
        # Windows ignores HOME for USERPROFILE. Passing the directory the hook
        # actually read makes the two agree by construction.
        self._run(
            "session-start",
            env={"HOME": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(REPO)},
            unset=("SOTA_ANCHOR_CACHE_DIR",),
        )
        cache = self._refresh_record()["cache"]
        assert cache.endswith(f"{tmp_path.name}/.cache/sota-anchor")

    def test_a_fresh_block_triggers_no_refresh(self, tmp_path):
        import time

        from sota_anchor.seed import SEED_FILENAME, write_seed

        write_seed("fresh block", path=tmp_path / SEED_FILENAME)
        self._run(
            "session-start",
            env={"SOTA_ANCHOR_CACHE_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(REPO)},
        )
        time.sleep(1.0)  # a detached refresh starts within milliseconds
        assert not self.refresh_log.exists()

    def test_prompt_hook_is_silent_by_default(self):
        result = self._run("user-prompt-submit", env={"SOTA_ANCHOR_PROMPT_HOOK": "0"},
                           stdin='{"prompt": "I cannot do this, add a workaround"}')
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_prompt_hook_nudges_on_a_limitation_claim_when_enabled(self):
        result = self._run("user-prompt-submit", env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
                           stdin='{"prompt": "The model cannot read vector data, add a workaround"}')
        payload = json.loads(result.stdout)
        assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "check-what-exists" in payload["hookSpecificOutput"]["additionalContext"]

    def test_prompt_hook_stays_quiet_on_an_ordinary_prompt(self):
        result = self._run("user-prompt-submit", env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
                           stdin='{"prompt": "rename this variable to total_count"}')
        assert result.stdout.strip() == ""

    def test_prompt_hook_never_returns_a_blocking_status(self):
        for prompt in ("I cannot do this", "rename a variable", ""):
            result = self._run("user-prompt-submit", env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
                               stdin=prompt)
            assert result.returncode == 0

    def _nudged(self, prompt: str) -> bool:
        result = self._run(
            "user-prompt-submit",
            env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
            stdin=json.dumps({"prompt": prompt}),
        )
        return bool(result.stdout.strip())

    def test_prompt_hook_nudges_on_an_unapostrophised_cant(self):
        # The exact prompt a live session declined to check.
        assert self._nudged(
            "i need to convert some nwds to glbs, write a custom parer for it "
            "since i cant use oda to read the files"
        )

    def test_prompt_hook_nudges_on_a_rewrite_from_scratch(self):
        assert self._nudged("let's just write our own reader from scratch")

    def test_prompt_hook_nudges_on_reverse_engineering(self):
        assert self._nudged("we'll reverse engineer the binary layout")

    def test_prompt_hook_matches_whole_words_only(self):
        # "cant" sits inside "significant"; substring matching would fire here.
        assert not self._nudged("this is a significant refactor of the applicant table")

    def test_prompt_hook_nudge_names_the_advice_path(self):
        # Measured: the nudge fired on "i cant use oda" and the host still
        # advised from memory, because the nudge only spoke of building.
        result = self._run(
            "user-prompt-submit",
            env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
            stdin=json.dumps({"prompt": "this cannot be done, add a workaround"}),
        )
        nudge = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "advising that no such tool exists" in nudge

    def test_prompt_hook_nudge_claims_no_authority(self):
        import re

        from .test_framing import OVERRIDE_PATTERNS

        result = self._run(
            "user-prompt-submit",
            env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
            stdin=json.dumps({"prompt": "this cannot be done, add a workaround"}),
        )
        nudge = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        for pattern in OVERRIDE_PATTERNS:
            assert not re.search(pattern, nudge, re.IGNORECASE), pattern

    def test_prompt_hook_emits_no_block_decision(self):
        result = self._run("user-prompt-submit", env={"SOTA_ANCHOR_PROMPT_HOOK": "1"},
                           stdin='{"prompt": "this is impossible, work around it"}')
        assert "decision" not in json.loads(result.stdout)["hookSpecificOutput"]


class TestMcpPortability:
    """`command: "sota-anchor"` needs the package on PATH, and a live session
    reported the server failing to connect for exactly that reason. Running it
    from the plugin's own checkout removes the manual install step.
    """

    def test_server_runs_from_the_plugin_directory(self):
        server = load(MCP_MANIFEST)["mcpServers"][MCP_SERVER]
        assert "${CLAUDE_PLUGIN_ROOT}" in " ".join(server["args"])

    def test_server_does_not_assume_the_cli_is_on_path(self):
        server = load(MCP_MANIFEST)["mcpServers"][MCP_SERVER]
        assert server["command"] != "sota-anchor"


class TestLineEndingResilience:
    """A fresh clone with core.autocrlf=true checked out bootstrap-block.md as
    CRLF, and every line of injected context arrived carrying a literal \r.
    The hook must not depend on the reader's git configuration. (The execution
    test for this lives in TestHookExecution, which fakes the refresh tools.)
    """

    def test_bootstrap_block_is_declared_lf_in_gitattributes(self):
        attributes = (REPO / ".gitattributes").read_text(encoding="utf-8")
        assert "hooks/bootstrap-block.md" in attributes

    def test_every_shipped_hook_file_is_declared_lf(self):
        attributes = (REPO / ".gitattributes").read_text(encoding="utf-8")
        for shipped in ("session-start", "user-prompt-submit", "run-hook.cmd", "bootstrap-block.md"):
            assert shipped in attributes, shipped


class TestSkillToolPriority:
    """A live run shelled out to `sota-anchor` and got exit 127: the package was
    in a project .venv, not on PATH. MCP succeeded, but the failure was noise.
    """

    def _body(self) -> str:
        return SKILL.read_text(encoding="utf-8")

    def test_names_the_mcp_tool_as_the_primary_path(self):
        body = self._body()
        primary = body.index("primary path")
        assert "check_what_exists" in body[:primary + 200]

    def test_mcp_appears_before_the_cli_fallback(self):
        body = self._body()
        assert body.index("check_what_exists(") < body.index("evidence --query")

    def test_labels_the_cli_as_a_fallback(self):
        assert "Fallback" in self._body()

    def test_fallback_checks_the_plugin_venv_before_calling(self):
        # The venv that exists is the plugin's own: the MCP server's `uv run`
        # creates it in the plugin checkout. A project-relative ./.venv only
        # holds sota-anchor if the user installed it there by hand.
        body = self._body()
        assert "${CLAUDE_PLUGIN_ROOT}/.venv/bin/sota-anchor" in body
        assert "${CLAUDE_PLUGIN_ROOT}/.venv/Scripts/sota-anchor.exe" in body

    @pytest.mark.parametrize(
        "path",
        [SKILL, COMMANDS / "sota-check.md", COMMANDS / "sota-sync.md"],
        ids=lambda p: p.name,
    )
    def test_no_uv_run_escapes_the_plugin_checkout(self, path):
        # A bare `uv run` resolves against the user's project: no sota-anchor
        # there, and a .venv plus uv.lock created in any project that has a
        # pyproject.toml. Claude Code substitutes ${CLAUDE_PLUGIN_ROOT} in skill
        # and command text, so every invocation can name the checkout.
        for line in path.read_text(encoding="utf-8").splitlines():
            if "uv run" in line:
                assert '--project "${CLAUDE_PLUGIN_ROOT}"' in line, line

    def test_check_command_follows_the_skill_tool_order(self):
        body = (COMMANDS / "sota-check.md").read_text(encoding="utf-8")
        assert "check-what-exists" in body
        assert "check_what_exists" in body
        assert "sota-anchor evidence --query" not in body

    def test_fallback_checks_the_system_path_too(self):
        assert "command -v sota-anchor" in self._body()

    def test_warns_about_the_exact_failure_seen(self):
        assert "127" in self._body()

    def test_an_unresolvable_cli_is_not_treated_as_evidence(self):
        body = self._body().lower()
        assert "not evidence that the limitation holds" in body

    def test_skill_asks_for_both_query_vectors(self):
        body = self._body()
        assert "domain_query" in body
        assert "capability_query" in body

    def test_skill_states_the_default_baseline(self):
        assert "assumption stands" in self._body().lower()

    def test_skill_carries_no_rhetorical_framing(self):
        assert "abandon work" not in self._body().lower()


class TestVersionSingleSource:
    """The version appeared in five places by hand. Any one could drift."""

    def test_package_exposes_a_version(self):
        import sota_anchor

        assert sota_anchor.__version__

    def test_plugin_manifest_matches_the_package(self):
        import sota_anchor

        assert load(PLUGIN_MANIFEST)["version"] == sota_anchor.__version__

    def test_cli_reports_the_package_version(self):
        from click.testing import CliRunner

        import sota_anchor
        from sota_anchor.cli import main

        assert sota_anchor.__version__ in CliRunner().invoke(main, ["--version"]).output

    def test_user_agent_carries_the_package_version(self):
        import sota_anchor
        from sota_anchor.retriever import USER_AGENT

        assert sota_anchor.__version__ in USER_AGENT

    def test_no_module_hardcodes_a_version_literal(self):
        import re

        for module in ("cli.py", "server.py", "retriever.py"):
            source = (REPO / "src" / "sota_anchor" / module).read_text(encoding="utf-8")
            assert not re.search(r'"\d+\.\d+\.\d+"', source), module


class TestNoPersonalContactInSource:
    """This repository is public. A personal address in source is scraped, and
    the User-Agent one was also transmitted to arXiv and GitHub on every request.
    Contact happens through the repository instead.
    """

    SOURCE_GLOBS = ("src/**/*.py", "*.toml", ".claude-plugin/*.json", ".mcp.json",
                    "hooks/*", "skills/**/*.md", "commands/*.md")

    def _tracked_text(self):
        import glob

        for pattern in self.SOURCE_GLOBS:
            for path in glob.glob(str(REPO / pattern), recursive=True):
                candidate = Path(path)
                if candidate.is_file():
                    try:
                        yield candidate, candidate.read_text(encoding="utf-8")
                    except UnicodeDecodeError:  # pragma: no cover - binary asset
                        continue

    def test_no_email_address_appears_in_source(self):
        import re

        pattern = re.compile(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}")
        offenders = [
            (path.name, match)
            for path, text in self._tracked_text()
            for match in pattern.findall(text)
            if "noreply@anthropic" not in match
        ]
        assert offenders == []

    def test_user_agent_identifies_the_client_by_repository(self):
        from sota_anchor.retriever import USER_AGENT

        assert "sota-anchor" in USER_AGENT
        assert "github.com" in USER_AGENT
        assert "mailto:" not in USER_AGENT
