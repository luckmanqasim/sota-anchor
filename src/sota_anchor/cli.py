"""`sota-anchor` command line interface."""

from __future__ import annotations

import asyncio
import json
import sys

import click

from .arbiter import (
    ArbiterError,
    LLMClient,
    LLMUnavailable,
    resolve_settings,
    verify_architecture,
)
from .catalog import (
    DEFAULT_MAX_PER_PROVIDER,
    DEFAULT_STALENESS_MONTHS,
    FOCUS_PROVIDERS,
    CatalogUnavailable,
    fetch_catalog,
)
from .injector import sync_targets
from .retriever import DEFAULT_WINDOW_MONTHS, gather_evidence
from .server import build_server

#: `check` exits 2 on an obsolete proposal so it can gate CI without parsing output.
EXIT_OBSOLETE = 2


def build_llm() -> LLMClient:
    return LLMClient(resolve_settings())


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252; keep output from dying on a code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - stream not reconfigurable
                pass


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0", prog_name="sota-anchor")
def main() -> None:
    """Break epistemic inertia: anchor agents to current models and live capabilities."""
    _force_utf8_output()


@main.command()
@click.option(
    "--target",
    type=click.Choice(["claude", "cursor", "agents", "all"]),
    default="claude",
    show_default=True,
    help="Which instruction files to refresh.",
)
@click.option(
    "--staleness-months",
    type=click.IntRange(min=1),
    default=DEFAULT_STALENESS_MONTHS,
    show_default=True,
    help="How far behind its own provider's newest release a model may be before it "
    "counts as legacy. Raise it for providers that ship slowly.",
)
@click.option(
    "--provider",
    "providers",
    multiple=True,
    help="Provider to feature in the block; repeatable. Defaults to "
    f"{', '.join(FOCUS_PROVIDERS)}.",
)
@click.option(
    "--all-providers",
    is_flag=True,
    help="Feature every provider. Costs roughly 4,600 tokens of resident context "
    "against roughly 510 for the default three.",
)
@click.option(
    "--max-per-provider",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_PER_PROVIDER,
    show_default=True,
    help="How many endpoints to list per provider.",
)
@click.option("--refresh", is_flag=True, help="Ignore the cache and refetch.")
def sync(
    target: str,
    staleness_months: int,
    providers: tuple[str, ...],
    all_providers: bool,
    max_per_provider: int,
    refresh: bool,
) -> None:
    """Fetch the live model catalog and refresh local agent instruction files."""
    try:
        catalog = asyncio.run(
            fetch_catalog(staleness_months=staleness_months, force=refresh)
        )
    except CatalogUnavailable as error:
        raise click.ClickException(str(error)) from error

    chosen = None if all_providers else (list(providers) or list(FOCUS_PROVIDERS))
    block = catalog.render_markdown(chosen, max_per_provider)

    if catalog.stale:
        click.echo("warning: served a cached catalog; the refresh attempt failed.", err=True)

    featured = catalog.featured(chosen, max_per_provider)
    click.echo(f"Catalog retrieved {catalog.fetched_at.date().isoformat()}: {len(featured)} active endpoints.")
    for result in sync_targets(target, block):
        state = "created" if result.created else ("updated" if result.changed else "unchanged")
        click.echo(f"  {state}: {result.path}")


@main.command()
@click.argument("proposal")
@click.option(
    "--months",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW_MONTHS,
    show_default=True,
    help="Recency window for evidence retrieval.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
def check(proposal: str, months: int, as_json: bool) -> None:
    """Test whether PROPOSAL relies on an obsolete limitation.

    Exits 0 when the limitation still holds, 2 when the proposal is obsolete.
    """
    try:
        llm = build_llm()
    except LLMUnavailable as error:
        raise click.ClickException(str(error)) from error

    try:
        report = asyncio.run(
            verify_architecture(proposal, llm=llm, gather=gather_evidence, months=months)
        )
    except (ArbiterError, CatalogUnavailable) as error:
        raise click.ClickException(str(error)) from error

    if as_json:
        click.echo(report.model_dump_json(indent=2))
    else:
        click.echo(report.render())

    if report.verdict.is_obsolete:
        raise SystemExit(EXIT_OBSOLETE)


@main.command()
def serve() -> None:
    """Run the MCP server over stdio for Claude Code, Cursor or OpenCode."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
