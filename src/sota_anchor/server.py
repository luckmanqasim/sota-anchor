"""MCP server exposing the catalog and the verification pipeline.

Every handler answers in prose rather than raising: a tool that throws gives the
calling agent a stack trace to interpret, while a tool that explains what is
missing gives it something to act on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from mcp.server import MCPServer

from . import __version__
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
from .protocol import build_verification_payload
from .retriever import DEFAULT_WINDOW_MONTHS, gather_evidence

SERVER_NAME = "sota-anchor"

VERIFY_DESCRIPTION = (
    "Check whether a plan relies on something being unavailable that may no longer "
    "be: no library or reader for a file format, only a vendor SDK able to open it, "
    "no API for the data, or a model that cannot do the task. Call it before "
    "writing a custom parser, converter or reader, rewriting something from "
    "scratch, or adding a workaround, heuristic pipeline or post-processing stage, "
    "in any domain. Call it with just the pitch first: it returns an inversion "
    "prompt asking what would have to be unavailable for the plan to be justified. "
    "Answer that, then call again with domain_query (the task in its own "
    "vocabulary) and capability_query (whatever would make the workaround "
    "unnecessary: an existing implementation, library, tool or model capability), "
    "each led by its most specific term, such as a format, product or library "
    "name. It returns dated evidence from papers, repositories and package "
    "registries, plus the standard for judging it: either the limitation still "
    "holds, or an assertion-reason update naming what replaces the workaround. "
    "Two queries because a narrow one misses a general advance indexed elsewhere. "
    "Raise months if a pass returns nothing."
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
        version=__version__,
        instructions=(
            "models://active lists the model API endpoints a public registry "
            "currently serves; read it before writing a model identifier into code "
            "or config. Call verify_architecture before building around something "
            "assumed to be unavailable - a custom parser or converter for a format "
            "thought to need vendor tooling, a from-scratch rewrite, or a "
            "workaround for a model limitation."
        ),
    )

    @mcp.resource("models://active", mime_type="application/json")
    async def active_models() -> str:
        """Model API endpoints a public registry currently serves, dated and sourced,
        with superseded identifiers mapped forward. It lists what can be called; it
        says nothing about which model is reading it."""
        try:
            catalog = await fetch_catalog()
        except CatalogUnavailable as error:
            return json.dumps({"error": str(error)})
        return json.dumps(catalog_payload(catalog), indent=2)

    @mcp.tool(description=VERIFY_DESCRIPTION)
    async def verify_architecture_tool(
        pitch: str,
        domain_query: str | None = None,
        capability_query: str | None = None,
        months: int = DEFAULT_WINDOW_MONTHS,
    ) -> str:
        try:
            llm = build_llm()
        except LLMUnavailable:
            # The default path, not a failure. With no provider account the
            # caller does the reasoning and this tool supplies the evidence and
            # the protocol, so the absence of a key is never reported as a fault.
            llm = None

        if llm is None:
            # The registry snapshot grounds the judge in dated fact about what
            # current endpoints declare. Its absence is not fatal.
            try:
                catalog = await fetch_catalog()
            except CatalogUnavailable:
                catalog = None
            try:
                payload = await build_verification_payload(
                    pitch,
                    domain_query=domain_query,
                    capability_query=capability_query,
                    gather=gather_evidence,
                    catalog=catalog,
                    months=months,
                )
            except ValueError as error:
                return f"sota-anchor cannot verify this proposal: {error}"
            return payload.render()

        try:
            report = await verify_architecture(
                pitch,
                llm=llm,
                gather=gather_evidence,
                months=months,
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

    @mcp.prompt(
        description="Project baseline: current model API endpoints plus the habit of "
        "checking assumed limitations."
    )
    async def init_project() -> str:
        try:
            catalog = await fetch_catalog()
            block = catalog.render_markdown(providers, max_per_provider)
        except CatalogUnavailable as error:
            block = f"(model catalog unavailable: {error})"

        return "\n".join(
            [
                "This project uses sota-anchor, which keeps a dated list of model API",
                "endpoints and checks designs against recent work.",
                "",
                block,
                "",
                "## Verifying assumptions",
                "",
                "Training data has a cutoff, and so do assumptions about what exists.",
                "Before building around something assumed to be unavailable - a custom",
                "parser or converter for a format thought to need vendor tooling, a",
                "rewrite of an existing library from scratch, an OCR or heuristic stage",
                "around a model limitation - call the `verify_architecture` tool with",
                "the design in question. It searches recent papers, repositories and",
                "package registries, and reports whether an existing solution already",
                "replaces what was about to be built.",
                "",
                "Read the `models://active` resource before writing a model identifier",
                "into code, config or a .env file.",
            ]
        )

    return mcp


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
