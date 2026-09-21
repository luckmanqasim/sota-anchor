"""The SessionStart context block.

Rendered at sync time and cached, so the hook that injects it is a `cat` rather
than a Python process: measured here, the pre-rendered path costs ~160 ms
against ~650 ms warm and ~1,090 ms cold through the interpreter.

The block is resident in every session the plugin touches, which sets its two
constraints. It has to be forceful and *specific* - naming the exact superseded
identifiers an agent would otherwise reach for, because a generic "use current
models" does not dislodge a confident prior. And it has to stay small, because
every token is paid on every session start.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

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


BOOTSTRAP_BLOCK = """[SOTA ANCHOR]

No model catalog has been fetched yet, so the current frontier lineup is not
available in this session.

Your training data has a cutoff, and model identifiers you remember are likely
to be superseded. Do not write a model identifier into code, config or a .env
file from memory. Ask the user which endpoint to use, or run `/sota-sync` to
fetch the live catalog.

Before building a workaround for something a tool "cannot" do, run
`/sota-check "<your design>"` - the limitation may have been removed since your
training cutoff.
"""


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
    """Condense the catalog into a session-start directive."""
    featured = catalog.featured(providers, max_per_provider)
    superseded = _select_forbidden(catalog, providers, max_per_provider, max_forbidden)
    fetched = catalog.fetched_at.date().isoformat()
    freshness = f"{fetched} (cached; last refresh failed)" if catalog.stale else fetched

    lines = [
        "[SOTA ANCHOR]",
        "",
        f"Active production model endpoints, retrieved {freshness}:",
        "",
    ]
    lines += [f"  {model.id}" for model in featured]
    # Counted from registry fields rather than asserted, so the claim cannot go
    # stale: an agent designing around "models only read and emit plain text" is
    # routing around something the registry already contradicts.
    accepts_images = sum(1 for m in featured if "image" in m.input_modalities)
    with_tools = sum(1 for m in featured if m.supports_tools)
    structured = sum(1 for m in featured if m.supports_structured_output)
    declared = [
        f"{accepts_images} accept image input" if accepts_images else "",
        f"{with_tools} support tool calls" if with_tools else "",
        f"{structured} support structured output" if structured else "",
    ]
    declared = [item for item in declared if item]

    lines += [
        "",
        "Your training data is older than that list. Where the two disagree, the",
        "list is right, and an identifier you do not recognise means the model is",
        "newer than you are - not that it is wrong.",
        "",
    ]
    if declared:
        summary = (
            f"The registry reports that of those {len(featured)}, "
            + ", ".join(declared)
            + ". Before designing around an assumption that a model can only "
            "read or emit plain text, check it against that."
        )
        lines += [*textwrap.wrap(summary, width=74), ""]
    lines += [
        "Do NOT write a superseded identifier into code, config, a .env file or a",
        "client default. In particular, do not reach for:",
        "",
    ]
    lines += [f"  {model_id}" for model_id in superseded]
    lines += [
        "",
        "If a task needs an endpoint not listed above, say so rather than guessing.",
        "",
        "Before building a workaround for something a tool \"cannot\" do - an OCR or",
        "parsing stage, a heuristic pipeline, a custom post-processing step - run",
        "`/sota-check \"<your design>\"`. The limitation may have been removed since",
        "your training cutoff.",
    ]
    return "\n".join(lines) + "\n"


def write_seed(block: str, *, path: Path | None = None) -> Path:
    """Write the block atomically as UTF-8, whatever the platform default is."""
    target = Path(path) if path is not None else seed_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
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
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        modified = dt.datetime.fromtimestamp(Path(path).stat().st_mtime, dt.timezone.utc)
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
