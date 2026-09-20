"""MCP server exposing the catalog and the verification pipeline.

Every handler answers in prose rather than raising: a tool that throws gives the
calling agent a stack trace to interpret, while a tool that explains what is
missing gives it something to act on.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Sequence

from mcp.server import MCPServer

from .arbiter import (
    ArbiterError,
    LLMClient,
    LLMUnavailable,
    resolve_settings,
    verify_architecture,
)
from .catalog import (
    DEFAULT_MAX_PER_PROVIDER,
    FOCUS_PROVIDERS,
    Catalog,
    CatalogUnavailable,
    fetch_catalog,
)
from .retriever import DEFAULT_WINDOW_MONTHS, gather_evidence

SERVER_NAME = "sota-anchor"

VERIFY_DESCRIPTION = (
    "Check whether a technical proposal relies on a limitation that recent tooling, "
    "frontier model capabilities or open-source work has already removed. Pass any "
    "architecture pitch, design note or task description. Returns either a "
    "confirmation that the limitation still holds, or an assertion-reason paradigm "
    "update telling you what not to build and what replaces it. Call this before "
    "committing to a workaround, a heuristic pipeline or a custom post-processing "
    "stage in any domain."
)


def build_llm() -> LLMClient:
    """Construct the LLM client from the environment."""
    return LLMClient(resolve_settings())


def catalog_payload(catalog: Catalog) -> dict[str, object]:
    return {
        "fetched_at": catalog.fetched_at.isoformat(),
        "source": catalog.source,
        "stale": catalog.stale,
        "staleness_months": catalog.staleness_months,
        "current": [
            {
                "id": model.id,
                "provider": model.provider,
                "version": model.version,
                "created": model.created.date().isoformat(),
                "context_length": model.context_length,
                "knowledge_cutoff": model.knowledge_cutoff,
            }
            for model in catalog.featured()
        ],
        "superseded": catalog.legacy_map(),
    }


def build_server(
    providers: Sequence[str] | None = FOCUS_PROVIDERS,
    max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
) -> MCPServer:
    mcp = MCPServer(
        SERVER_NAME,
        version="0.1.0",
        instructions=(
            "Consult models://active before writing any model identifier, and call "
            "verify_architecture before committing to a workaround for a claimed "
            "tool limitation."
        ),
    )

    @mcp.resource("models://active", mime_type="application/json")
    async def active_models() -> str:
        """Current production model endpoints, with superseded ones mapped forward."""
        try:
            catalog = await fetch_catalog()
        except CatalogUnavailable as error:
            return json.dumps({"error": str(error)})
        return json.dumps(catalog_payload(catalog), indent=2)

    @mcp.tool(description=VERIFY_DESCRIPTION)
    async def verify_architecture_tool(pitch: str) -> str:
        try:
            llm = build_llm()
        except LLMUnavailable as error:
            return f"sota-anchor cannot verify this proposal: {error}"

        try:
            report = await verify_architecture(
                pitch,
                llm=llm,
                gather=gather_evidence,
                months=DEFAULT_WINDOW_MONTHS,
            )
        except (ArbiterError, CatalogUnavailable) as error:
            return f"sota-anchor could not complete verification: {error}"

        return report.render()

    # Registered under the name the spec gives, without shadowing the imported
    # pipeline function inside this module.
    mcp.remove_tool("verify_architecture_tool")
    mcp.add_tool(
        verify_architecture_tool,
        name="verify_architecture",
        description=VERIFY_DESCRIPTION,
    )

    @mcp.prompt(description="Baseline system prompt: active models plus verification habit.")
    async def init_project() -> str:
        try:
            catalog = await fetch_catalog()
            block = catalog.render_markdown(providers, max_per_provider)
        except CatalogUnavailable as error:
            block = f"(model catalog unavailable: {error})"

        return "\n".join(
            [
                "You are working in a project anchored by sota-anchor.",
                "",
                block,
                "",
                "## Verifying assumptions",
                "",
                "Your training data has a cutoff. Before committing to a workaround,",
                "a heuristic pipeline, an OCR or parsing stage, or any custom",
                "post-processing that exists because a tool 'cannot' do something,",
                "call the `verify_architecture` tool with the design in question.",
                "It checks the assumption against research and repositories from the",
                "past 12 months and tells you if a native primitive already replaces",
                "what you were about to build.",
                "",
                "Read the `models://active` resource before writing any model",
                "identifier into code, config or a .env file.",
            ]
        )

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
