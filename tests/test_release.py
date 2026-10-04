"""Release metadata agrees wherever a version or the registry name is written.

The package, the plugin and the MCP Registry entry each state the version, and a
release that bumps one without the others ships two answers to "which version is
this". The registry also refuses to list a PyPI package whose description lacks
the entry's name.
"""

import json
import re
import tomllib
from pathlib import Path

from sota_anchor.retriever import GITHUB_AUTH_ENVS, WEB_API_KEY_ENV

REPO = Path(__file__).resolve().parent.parent


def project() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def registry_entry() -> dict:
    return json.loads((REPO / "server.json").read_text(encoding="utf-8"))


class TestReleaseMetadata:
    def test_plugin_version_matches_the_package(self):
        plugin = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
        assert plugin["version"] == project()["version"]

    def test_registry_entry_matches_the_package(self):
        entry = registry_entry()
        [package] = entry["packages"]
        assert entry["version"] == package["version"] == project()["version"]
        assert package["registryType"] == "pypi"
        assert package["identifier"] == project()["name"]

    def test_registry_text_fits_its_limits(self):
        # The registry's schema caps both at 100 characters and refuses the entry otherwise.
        entry = registry_entry()
        assert 0 < len(entry["description"]) <= 100
        assert 0 < len(entry["title"]) <= 100

    def test_the_readme_names_the_registry_entry(self):
        # PyPI's copy of the README is what the registry reads before listing.
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        assert f"<!-- mcp-name: {registry_entry()['name']} -->" in readme

    def test_the_registry_starts_the_server(self):
        [package] = registry_entry()["packages"]
        assert package["transport"] == {"type": "stdio"}
        assert [argument["value"] for argument in package["packageArguments"]] == ["serve"]

    def test_the_registry_offers_the_keys_the_code_reads(self):
        [package] = registry_entry()["packages"]
        offered = {variable["name"]: variable for variable in package["environmentVariables"]}
        assert set(offered) == {WEB_API_KEY_ENV, *GITHUB_AUTH_ENVS}
        for variable in offered.values():
            assert variable["isSecret"] is True
            assert variable["isRequired"] is False


class TestPublishWorkflow:
    """The job that can mint a PyPI token is the one an attacker would want to run
    code in, so it runs none of the project's, and every action is a fixed commit."""

    WORKFLOW = REPO / ".github" / "workflows" / "publish.yml"

    def _text(self) -> str:
        return self.WORKFLOW.read_text(encoding="utf-8")

    def _job(self, name: str) -> str:
        """The lines of one job, up to the next key at the same indent."""
        lines = self._text().splitlines()
        first = lines.index(f"  {name}:")
        rest = [i for i in range(first + 1, len(lines)) if re.fullmatch(r"  [a-z][a-z0-9_-]*:", lines[i])]
        return chr(10).join(lines[first : rest[0] if rest else len(lines)])

    def test_every_action_is_pinned_to_a_commit(self):
        uses = re.findall(r"uses:\s*(\S+)", self._text())
        assert uses
        assert [u for u in uses if not re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u)] == []

    def test_only_the_publishing_job_can_mint_a_token(self):
        assert self._text().count("id-token: write") == 1
        assert "id-token: write" in self._job("pypi")

    def test_the_publishing_job_runs_no_project_code(self):
        job = self._job("pypi")
        for step in ("checkout", "uv sync", "uv run", "pytest", "uv build"):
            assert step not in job

    def test_the_publishing_job_uses_the_registered_environment(self):
        assert "environment: pypi" in self._job("pypi")

    def test_only_a_release_publishes(self):
        lines = self._text().splitlines()
        triggers: list[str] = []
        for line in lines[lines.index("on:") + 1 :]:
            if line and not line.startswith(" "):
                break
            triggers.append(line.strip())
        assert "release:" in triggers
        assert not any(t.startswith("workflow_dispatch") for t in triggers)
