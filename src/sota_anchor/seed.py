"""The SessionStart context block.

Rendered at sync time and cached, so the hook that injects it is a `cat` rather
than a Python process: measured here, the pre-rendered path costs ~160 ms
against ~650 ms warm and ~1,090 ms cold through the interpreter.

The block is resident in every session the plugin touches, which sets its
constraints. It has to be *specific* - naming the exact superseded identifiers
an agent would otherwise reach for, because a generic "use current models" does
not dislodge a confident prior. It has to stay small, because every token is
paid on every session start.

And it has to read as what it is: sourced reference data from a plugin the user
installed. An earlier version asserted that "where the two disagree, the list
is right" and that an unrecognised identifier "means the model is newer than
you are", under the heading "Active production model endpoints". A host model
read that as a claim about which model was running plus an order to override
its own knowledge, and refused the whole block as a prompt injection. So the
block now says where the data came from and when, says what it is for, and
claims no authority over the reader.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import textwrap
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from .catalog import (
    DEFAULT_MAX_PER_PROVIDER,
    FOCUS_PROVIDERS,
    Catalog,
    default_cache_path,
    fetch_catalog,
)

SEED_FILENAME = "session-block.md"
SEED_TTL_HOURS = 24

#: How many superseded identifiers to name explicitly. Enough to catch the ones
#: an agent actually reaches for, few enough to stay cheap.
MAX_FORBIDDEN = 10


#: Width the prose is wrapped to. The block is read by a model, not a terminal,
#: but short lines keep the cached file reviewable by a human too.
WRAP_WIDTH = 76

#: Closes both blocks. Written out once so the bootstrap and the rendered block
#: cannot describe the check differently.
CHECK_PARAGRAPH = (
    "Before building around something assumed to be unavailable - a custom "
    "parser or converter for a format thought to need vendor tooling, a rewrite "
    "of an existing library from scratch, an OCR or heuristic stage around a "
    "model limitation - or before advising that no open-source tool exists for "
    'it, run `/sota-check "<the design>"` or the check_what_exists MCP tool. '
    "It searches recent papers, repositories and package registries for an "
    "existing solution first."
)


def _paragraph(text: str) -> list[str]:
    return textwrap.wrap(text, width=WRAP_WIDTH)


BOOTSTRAP_BLOCK = "\n".join(
    [
        "sota-anchor plugin: no model registry snapshot yet",
        "",
        *_paragraph(
            "No registry snapshot has been fetched, so this session has no current "
            "list of model API endpoints. Model identifiers recalled from training "
            "data may have been superseded since. When code or config needs one, "
            "ask the user which endpoint to use, or run /sota-sync to fetch the "
            "current list."
        ),
        "",
        *_paragraph(CHECK_PARAGRAPH),
    ]
) + "\n"


def seed_path() -> Path:
    return default_cache_path().parent / SEED_FILENAME


def _select_forbidden(
    catalog: Catalog,
    providers: Sequence[str] | None,
    max_per_provider: int,
    limit: int,
) -> list[str]:
    """Pick which superseded identifiers to name, spread across providers.

    Taking the legacy map in ID order put every slot on one provider on live
    data, truncating away `openai/gpt-4o` and `google/gemini-2.5-flash` - the
    identifiers an older-cutoff model is most likely to emit.

    Within a provider it takes **one entry per lineage** - the newest superseded
    member, which is the model the current one directly replaced - and prefers
    lineages that still have a current model. Ranking by age instead surfaced
    `gpt-3.5-turbo-instruct` and `gemma-2-27b-it` on live data while pushing out
    `gpt-4o` and `gemini-2.5-flash`, and listing a whole version run
    (opus-4.1 through opus-4.8) spends resident tokens on one family.
    """
    # Drawn from every superseded model, not from `legacy_map`. That map only
    # contains entries whose successor is itself featured, which on live data
    # dropped `openai/gpt-4o` because `gpt-5.5` missed the featured cut. The block
    # renders no mapping - only identifiers not to use - so it needs no successor.
    wanted = set(providers) if providers is not None else None
    featured_lineages = {
        (m.provider, m.lineage) for m in catalog.featured(providers, max_per_provider)
    }

    # Two candidates per lineage: the model the current one directly replaced,
    # and the oldest superseded member of the same family. The first is what a
    # recently-trained agent reaches for; the second is the training-era name the
    # brief asks to forbid by example (gpt-4o, gemini-2.5, claude-3.5). Keeping
    # only one let gpt-5.4 shadow gpt-4o inside the (gpt,) lineage on live data.
    newest: dict[tuple[str, tuple[str, ...]], Any] = {}
    oldest: dict[tuple[str, tuple[str, ...]], Any] = {}
    for entry in catalog.models:
        if entry.status not in ("legacy", "retired") or entry.is_alias:
            continue
        if wanted is not None and entry.provider not in wanted:
            continue
        key = (entry.provider, entry.lineage)
        if key not in newest or entry.created > newest[key].created:
            newest[key] = entry
        if key not in oldest or entry.created < oldest[key].created:
            oldest[key] = entry

    def tier(key: tuple[str, tuple[str, ...]]) -> int:
        """Lineages with a current model first; a dead-end family is a wasted slot."""
        return 0 if key in featured_lineages else 1

    ranked: dict[str, list[str]] = {}
    for key in sorted(newest, key=lambda k: (tier(k), -newest[k].created.timestamp(), k)):
        provider = key[0]
        slots = ranked.setdefault(provider, [])
        for candidate in (newest[key].id, oldest[key].id):
            if candidate not in slots:
                slots.append(candidate)

    chosen: list[str] = []
    for position in range(max(len(ids) for ids in ranked.values()) if ranked else 0):
        for provider in sorted(ranked):
            if position < len(ranked[provider]) and len(chosen) < limit:
                chosen.append(ranked[provider][position])
        if len(chosen) >= limit:
            break
    return chosen


def render_seed(
    catalog: Catalog,
    providers: Sequence[str] | None = FOCUS_PROVIDERS,
    max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
    max_forbidden: int = MAX_FORBIDDEN,
) -> str:
    """Condense the catalog into sourced, scoped reference text for a session."""
    featured = catalog.featured(providers, max_per_provider)
    superseded = _select_forbidden(catalog, providers, max_per_provider, max_forbidden)
    fetched = catalog.fetched_at.date().isoformat()
    freshness = f"{fetched} (a cached copy; the last refresh failed)" if catalog.stale else fetched

    lines = [
        "sota-anchor plugin: model registry snapshot",
        "",
        *_paragraph(
            f"Retrieved {freshness} from {catalog.source}. It lists the model API "
            "endpoints served there, for use when code or config needs a model "
            "identifier. It says nothing about which model is running this "
            "session."
        ),
        "",
    ]

    if not featured:
        lines += [
            *_paragraph(
                "The snapshot has no endpoints for the selected providers, so it "
                "cannot supply a current identifier. When code or config needs one, "
                "ask the user which endpoint to use."
            ),
            "",
        ]
    else:
        heading = "Current endpoint:" if len(featured) == 1 else "Current endpoints:"
        lines += [heading, "", *(f"  {model.id}" for model in featured), ""]
        summary = _declared_interfaces(featured)
        if summary:
            lines += [*_paragraph(summary), ""]

    if superseded:
        lines += [
            *_paragraph(
                "Superseded identifiers - the same provider now serves newer models. "
                "Prefer a current endpoint over these when writing code, config or a "
                ".env file:"
            ),
            "",
            *(f"  {model_id}" for model_id in superseded),
            "",
        ]

    if featured:
        lines += [
            *_paragraph(
                "Every current endpoint above was being served on that date, "
                "including any released after a model's training data was "
                "collected. If a task needs an endpoint that is not listed, say so "
                "rather than guessing one."
            ),
            "",
        ]

    lines += _paragraph(CHECK_PARAGRAPH)
    return "\n".join(lines) + "\n"


def _declared_interfaces(featured: Sequence[Any]) -> str:
    """What the listed endpoints declare, counted from registry fields.

    Counted rather than asserted, so the sentence cannot go stale: an agent
    designing around "models only read and emit plain text" is routing around
    something the registry already contradicts.
    """
    accepts_images = sum(1 for m in featured if "image" in m.input_modalities)
    with_tools = sum(1 for m in featured if m.supports_tools)
    structured = sum(1 for m in featured if m.supports_structured_output)
    if not (accepts_images or with_tools or structured):
        return ""

    checkable = (
        " A design premised on models reading or emitting only plain text can be "
        "checked against that."
    )
    if len(featured) == 1:
        declared = [
            "accepts image input" if accepts_images else "",
            "supports tool calls" if with_tools else "",
            "supports structured output" if structured else "",
        ]
        return f"The registry reports that it {_join([d for d in declared if d])}." + checkable

    declared = [
        f"{accepts_images} accept image input" if accepts_images else "",
        f"{with_tools} support tool calls" if with_tools else "",
        f"{structured} support structured output" if structured else "",
    ]
    return (
        f"Of these {len(featured)}, the registry reports that "
        f"{_join([d for d in declared if d])}." + checkable
    )


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def write_seed(block: str, *, path: Path | None = None) -> Path:
    """Write the block atomically as UTF-8, whatever the platform default is."""
    target = Path(path) if path is not None else seed_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # delete=False is required: the file must be closed before os.replace
    # can move it into place atomically. `with handle:` below closes it.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=target.name,
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(block)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def is_stale(path: Path, *, now: dt.datetime | None = None, ttl_hours: int = SEED_TTL_HOURS) -> bool:
    """True when the block is missing or older than the TTL.

    The hook makes the same judgement in pure bash with `find -mmin`; this is the
    Python-side equivalent for the CLI and tests.
    """
    now = now or dt.datetime.now(dt.UTC)
    try:
        modified = dt.datetime.fromtimestamp(Path(path).stat().st_mtime, dt.UTC)
    except OSError:
        return True
    return modified + dt.timedelta(hours=ttl_hours) <= now


CatalogFetcher = Callable[..., Awaitable[Catalog]]


async def refresh(
    *,
    fetch: CatalogFetcher | None = None,
    seed_file: Path | None = None,
    providers: Sequence[str] | None = FOCUS_PROVIDERS,
    max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
    **fetch_kwargs: Any,
) -> Path:
    """Refetch the catalog and rewrite the cached block.

    A catalog failure propagates without touching the existing block: a stale
    block is worth far more than no block, since the hook would otherwise fall
    back to the bootstrap text and the session would lose the model list entirely.
    """
    fetch = fetch or fetch_catalog
    catalog = await fetch(**fetch_kwargs)
    return write_seed(
        render_seed(catalog, providers, max_per_provider), path=seed_file
    )
