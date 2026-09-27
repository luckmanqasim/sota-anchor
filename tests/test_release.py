"""Release metadata agrees wherever a version or the registry name is written.

The package, the plugin and the MCP Registry entry each state the version, and a
release that bumps one without the others ships two answers to "which version is
this". The registry also refuses to list a PyPI package whose description lacks
the entry's name.
"""

import json
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
