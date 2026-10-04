"""`sota-anchor` command line interface."""

from __future__ import annotations

import asyncio
import contextlib
import sys

import click

from . import __version__
from .arbiter import (
    ArbiterError,
    LLMClient,
    LLMUnavailable,
    check_what_exists,
    resolve_settings,
)
from .catalog import (
    DEFAULT_MAX_PER_PROVIDER,
    DEFAULT_STALENESS_MONTHS,
    FOCUS_PROVIDERS,
    CatalogUnavailable,
    fetch_catalog,
)
from .injector import sync_targets
from .protocol import build_verification_payload
from .retriever import DEFAULT_WINDOW_MONTHS, gather_evidence
from .seed import render_seed, seed_path, write_seed
from .server import build_server

#: `check` exits 2 on an obsolete proposal so it can gate CI without parsing output.
EXIT_OBSOLETE = 2

#: And 3 when no verdict was reached, because a host model still has to judge.
#: Collapsing this into 0 would let a gate read "nobody judged this" as "fine".
EXIT_INDETERMINATE = 3


def build_llm() -> LLMClient:
    return LLMClient(resolve_settings())


async def _keyless_payload(
    proposal: str, domain_query: str | None, capability_query: str | None, months: int
):
    """Build the host-facing payload, grounding it in the registry when reachable."""
    try:
        catalog = await fetch_catalog()
    except Exception:
        # The snapshot is optional grounding; its failure never stops the check.
        catalog = None
    return await build_verification_payload(
        proposal,
        domain_query=domain_query,
        capability_query=capability_query,
        gather=gather_evidence,
        catalog=catalog,
        months=months,
    )


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252; keep output from dying on a code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A stream that cannot be reconfigured is not worth failing over.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sota-anchor")
def main() -> None:
    """Find what shipped after your coding agent's training, before it rebuilds it from scratch.

    Searches recent papers, repositories and packages when a plan assumes something
    isn't available, and keeps a dated list of the model API IDs a public registry serves.
    """
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

    # Refresh what the SessionStart hook injects, so a sync updates both the
    # files a human reads and the block every future session starts with.
    write_seed(render_seed(catalog, chosen, max_per_provider), path=seed_path())
    for result in sync_targets(target, block):
        state = "created" if result.created else ("updated" if result.changed else "unchanged")
        click.echo(f"  {state}: {result.path}")


@main.command()
@click.argument("proposal")
@click.option(
    "--domain-query",
    default=None,
    help="Task keywords in the domain's own vocabulary, from the inversion step. "
    "Supplying either query moves the keyless protocol to its second phase.",
)
@click.option(
    "--capability-query",
    default=None,
    help="Keywords for the broad capability that would make the workaround "
    "unnecessary. Searched alongside --domain-query, since a narrow query misses "
    "a general advance indexed under other terminology.",
)
@click.option(
    "--months",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW_MONTHS,
    show_default=True,
    help="Recency window for evidence retrieval.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
def check(
    proposal: str,
    domain_query: str | None,
    capability_query: str | None,
    months: int,
    as_json: bool,
) -> None:
    """Test whether PROPOSAL relies on an obsolete limitation.

    With an API key configured this runs the whole pipeline and exits 0 when the
    limitation still holds or 2 when the proposal is obsolete. Without one it
    prints the protocol for a host agent to answer and exits 3.
    """
    try:
        llm = build_llm()
    except LLMUnavailable as error:
        # Not a failure any more. The plugin's whole point is that the host
        # agent can do this reasoning, so hand it the protocol instead.
        try:
            payload = asyncio.run(
                _keyless_payload(proposal, domain_query, capability_query, months)
            )
        except ValueError as bad_input:
            raise click.ClickException(str(bad_input)) from bad_input
        click.echo(payload.render())
        click.echo(
            f"(no verdict rendered here: {error}. Set a key for a headless verdict, "
            "or let your agent answer the protocol above.)",
            err=True,
        )
        raise SystemExit(EXIT_INDETERMINATE) from error

    try:
        report = asyncio.run(
            check_what_exists(proposal, llm=llm, gather=gather_evidence, months=months)
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
@click.option("--query", required=True, help="Verification query to retrieve evidence for.")
@click.option(
    "--months",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW_MONTHS,
    show_default=True,
    help="Recency window.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the evidence set as JSON.")
def evidence(query: str, months: int, as_json: bool) -> None:
    """Retrieve recent evidence for QUERY. Deterministic, and needs no API key.

    This is what the check-what-exists skill calls: the plugin does the retrieval,
    the host agent does the judging.
    """
    result = asyncio.run(gather_evidence(query, months=months))
    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    click.echo(result.render())
    for error in result.errors:
        click.echo(f"warning: {error}", err=True)


@main.command()
@click.option("--refresh", is_flag=True, help="Refetch the catalog and rewrite the cache.")
@click.option(
    "--staleness-months",
    type=click.IntRange(min=1),
    default=DEFAULT_STALENESS_MONTHS,
    show_default=True,
    help="Legacy threshold, as for `sync`.",
)
def seed(refresh: bool, staleness_months: int) -> None:
    """Print the session-start context block, or refresh the cached copy.

    The SessionStart hook reads the cached file rather than calling this, so that
    injecting context costs a `cat` and not an interpreter start.
    """
    try:
        catalog = asyncio.run(
            fetch_catalog(staleness_months=staleness_months, force=refresh)
        )
    except CatalogUnavailable as error:
        raise click.ClickException(str(error)) from error

    block = render_seed(catalog)
    if refresh:
        target = write_seed(block, path=seed_path())
        click.echo(f"wrote {target}", err=True)
    click.echo(block)


@main.command()
def serve() -> None:
    """Run the MCP server over stdio for Claude Code, Cursor or OpenCode."""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
