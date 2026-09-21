"""Channel 1 — dynamic entity catalog.

Tiering is derived from two things only: the *structure* of a model ID and the
timestamps the registry reports. No model name, family or version is written
into this module, so a provider can rename or renumber its lineup without any
code change here.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
from pydantic import BaseModel, Field

CATALOG_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_HOURS = 24
DEFAULT_STALENESS_MONTHS = 12
DEFAULT_MAX_PER_PROVIDER = 4

#: Display filter for the injected block, not capability knowledge. Measured
#: context cost is ~510 tokens for these three against ~4,600 for all 61
#: providers, and the block is resident in every agent session.
FOCUS_PROVIDERS: tuple[str, ...] = ("anthropic", "google", "openai")

#: OpenRouter marks floating aliases (e.g. ``~google/gemini-pro-latest``) with a
#: leading sigil. They stay in the catalog but are never featured: an agent
#: should commit a concrete endpoint, not a registry-specific moving target.
ALIAS_SIGIL = "~"

Status = Literal["current", "legacy", "retired"]


class CatalogUnavailable(RuntimeError):
    """Raised when the registry is unreachable and no cache exists to fall back on."""


def default_cache_path() -> Path:
    return Path.home() / ".cache" / "sota-anchor" / "catalog.json"


class ParsedId(BaseModel):
    """A model ID decomposed into the parts tiering needs."""

    raw: str
    base_id: str
    provider: str
    lineage: tuple[str, ...]
    version: str | None
    variant: str | None
    is_alias: bool


def parse_model_id(model_id: str) -> ParsedId:
    """Split ``provider/slug[:variant]`` into lineage and version tokens.

    A slug token containing a digit is a *version* token; a purely alphabetic
    token belongs to the *lineage*. This is what lets ``gpt-4o`` and ``gpt-5.5``
    share the ``(gpt,)`` lineage across a naming-scheme change, so supersession
    still resolves between them.
    """
    base_id, _, variant = model_id.partition(":")
    provider, _, slug = base_id.partition("/")
    tokens = [token for token in slug.split("-") if token]
    lineage = tuple(t for t in tokens if not any(c.isdigit() for c in t))
    versions = [t for t in tokens if any(c.isdigit() for c in t)]
    return ParsedId(
        raw=model_id,
        base_id=base_id,
        provider=provider,
        lineage=lineage,
        version=versions[0] if versions else None,
        variant=variant or None,
        is_alias=provider.startswith(ALIAS_SIGIL),
    )


class ModelEntry(BaseModel):
    id: str
    name: str
    provider: str
    lineage: tuple[str, ...]
    version: str | None = None
    created: dt.datetime
    status: Status
    superseded_by: str | None = None
    knowledge_cutoff: str | None = None
    context_length: int | None = None
    prompt_price: float | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    supports_tools: bool = False
    supports_structured_output: bool = False
    is_alias: bool = False
    variants: tuple[str, ...] = ()

    @property
    def is_text_endpoint(self) -> bool:
        """Usable as a coding-agent endpoint, judged from registry data alone.

        Text-only output plus tool support is what separates a production
        endpoint from an image, audio or completion-only model. Deriving it from
        these fields avoids a hand-maintained list of model names.
        """
        return tuple(self.output_modalities) == ("text",) and self.supports_tools


class Catalog(BaseModel):
    fetched_at: dt.datetime
    source: str = CATALOG_URL
    staleness_months: int = DEFAULT_STALENESS_MONTHS
    stale: bool = False
    models: list[ModelEntry] = Field(default_factory=list)

    def by_id(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    def featured(
        self,
        providers: Sequence[str] | None = FOCUS_PROVIDERS,
        max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
    ) -> list[ModelEntry]:
        """Current, tool-capable, text-output models, flagship first, capped per provider.

        Pass ``providers=None`` for every provider in the registry.
        """
        eligible = [
            m
            for m in self.models
            if m.status == "current" and m.is_text_endpoint and not m.is_alias
        ]
        if providers is not None:
            wanted = set(providers)
            eligible = [m for m in eligible if m.provider in wanted]

        # Capability first, recency as tiebreak. Price is the only fully dynamic
        # capability signal the registry offers, and ranking by recency alone let a
        # newer-but-cheaper model (gemma-4-26b at $0.09/1M) push a provider's
        # flagship (gemini-3.1-pro at $2.00/1M) out of the block entirely.
        kept: list[ModelEntry] = []
        seen: dict[str, int] = {}
        ordered = sorted(
            eligible,
            key=lambda m: (m.provider, -(m.prompt_price or 0.0), -m.created.timestamp(), m.id),
        )
        for model in ordered:
            if seen.get(model.provider, 0) >= max_per_provider:
                continue
            seen[model.provider] = seen.get(model.provider, 0) + 1
            kept.append(model)
        return kept

    def legacy_map(
        self,
        providers: Sequence[str] | None = FOCUS_PROVIDERS,
        max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
    ) -> dict[str, str]:
        """Superseded ID -> the featured endpoint that replaces it."""
        featured_ids = {m.id for m in self.featured(providers, max_per_provider)}
        return {
            m.id: m.superseded_by
            for m in sorted(self.models, key=lambda m: m.id)
            if m.status in ("legacy", "retired")
            and m.superseded_by in featured_ids
        }

    def render_markdown(
        self,
        providers: Sequence[str] | None = FOCUS_PROVIDERS,
        max_per_provider: int = DEFAULT_MAX_PER_PROVIDER,
    ) -> str:
        """The body of the managed instruction block."""
        featured = self.featured(providers, max_per_provider)
        retired = self.legacy_map(providers, max_per_provider)
        fetched = self.fetched_at.date().isoformat()

        lines = [
            "## Active model endpoints",
            "",
            f"Retrieved from {self.source} on {fetched}"
            + (" (cached copy; refresh failed)" if self.stale else "")
            + ".",
            "",
            "Your training data is older than this table. When it disagrees with",
            "the table, the table is right.",
            "",
            "| Endpoint | Version | Context | Knowledge cutoff |",
            "| --- | --- | --- | --- |",
        ]
        for model in featured:
            context = f"{model.context_length:,}" if model.context_length else "-"
            lines.append(
                f"| `{model.id}` | {model.version or '-'} | {context} "
                f"| {model.knowledge_cutoff or '-'} |"
            )

        if retired:
            lines += [
                "",
                "### Superseded - do not scaffold these",
                "",
                "| Retired endpoint | Use instead |",
                "| --- | --- |",
            ]
            lines += [f"| `{old}` | `{new}` |" for old, new in retired.items()]

        lines += [
            "",
            "### Rules",
            "",
            "1. When writing a `.env`, API client, config default or model constant,",
            "   use an endpoint from the active table above. Never a superseded one.",
            "2. Do not \"correct\" an endpoint above to a name you recognise from",
            "   training. An unfamiliar name means the model is newer than you are.",
            "3. If a task needs an endpoint not listed here, say so rather than",
            "   guessing an identifier.",
        ]
        return "\n".join(lines)


def _shift_months(moment: dt.datetime, months: int) -> dt.datetime:
    """Calendar-accurate subtraction, so the window means what a human means."""
    total = (moment.year * 12 + moment.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def _as_datetime(value: Any) -> dt.datetime | None:
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _collapse_variants(raw_models: list[dict]) -> dict[str, tuple[dict, list[str]]]:
    """Fold ``:suffix`` rows onto their base ID, keeping the suffixes as metadata."""
    collapsed: dict[str, tuple[dict, list[str]]] = {}
    for raw in raw_models:
        model_id = str(raw.get("id") or "")
        if not model_id:
            continue
        parsed = parse_model_id(model_id)
        record, variants = collapsed.get(parsed.base_id, (None, []))
        if parsed.variant:
            variants = [*variants, parsed.variant]
        if record is None or (raw.get("created") or 0) > (record.get("created") or 0):
            record = raw
        collapsed[parsed.base_id] = (record, variants)
    return collapsed


def build_catalog(
    raw_models: list[dict],
    *,
    now: dt.datetime,
    staleness_months: int = DEFAULT_STALENESS_MONTHS,
    source: str = CATALOG_URL,
    stale: bool = False,
) -> Catalog:
    """Tier a raw registry payload.

    A model is ``legacy`` when a newer model shares its ``(provider, lineage)``
    group, or when it trails its *own* provider's frontier by more than the
    staleness window. The second rule is what demotes orphan lineages such as
    ``openai/gpt-4-turbo``, which the first would leave "current" in a lineage
    of one. Measuring against the provider's own frontier keeps a slow-shipping
    provider's newest model current.
    """
    collapsed = _collapse_variants(raw_models)

    entries: list[ModelEntry] = []
    for base_id, (raw, variants) in collapsed.items():
        parsed = parse_model_id(base_id)
        created = _as_datetime(raw.get("created")) or dt.datetime.min.replace(
            tzinfo=dt.timezone.utc
        )
        architecture = raw.get("architecture") or {}
        pricing = raw.get("pricing") or {}
        try:
            prompt_price = float(pricing.get("prompt")) if pricing.get("prompt") else None
        except (TypeError, ValueError):
            prompt_price = None

        entries.append(
            ModelEntry(
                id=base_id,
                name=str(raw.get("name") or base_id),
                provider=parsed.provider,
                lineage=parsed.lineage,
                version=parsed.version,
                created=created,
                status="current",
                knowledge_cutoff=raw.get("knowledge_cutoff") or None,
                context_length=raw.get("context_length"),
                prompt_price=prompt_price,
                input_modalities=tuple(architecture.get("input_modalities") or ()),
                output_modalities=tuple(architecture.get("output_modalities") or ()),
                supports_tools="tools" in (raw.get("supported_parameters") or []),
                supports_structured_output=bool(
                    {"structured_outputs", "response_format"}
                    & set(raw.get("supported_parameters") or [])
                ),
                is_alias=parsed.is_alias,
                variants=tuple(sorted(set(variants))),
            )
        )

    frontier: dict[str, dt.datetime] = {}
    for entry in entries:
        known = frontier.get(entry.provider)
        if known is None or entry.created > known:
            frontier[entry.provider] = entry.created

    groups: dict[tuple[str, tuple[str, ...]], list[ModelEntry]] = {}
    for entry in entries:
        groups.setdefault((entry.provider, entry.lineage), []).append(entry)

    expirations = {
        base_id: _as_datetime((raw.get("expiration_date")))
        for base_id, (raw, _) in collapsed.items()
    }

    for (provider, _lineage), group in groups.items():
        group.sort(key=lambda m: m.created, reverse=True)
        cutoff = _shift_months(frontier[provider], staleness_months)
        for rank, entry in enumerate(group):
            expires = expirations.get(entry.id)
            if expires is not None and expires < now:
                entry.status = "retired"
                entry.superseded_by = group[0].id if rank > 0 else None
                continue
            if rank > 0:
                entry.status = "legacy"
                entry.superseded_by = group[0].id
                continue
            if entry.created < cutoff:
                entry.status = "legacy"
                continue
            entry.status = "current"

    entries.sort(key=lambda m: (m.provider, -m.created.timestamp(), m.id))
    return Catalog(
        fetched_at=now,
        source=source,
        staleness_months=staleness_months,
        stale=stale,
        models=entries,
    )


def _read_cache(cache_path: Path) -> tuple[list[dict], dt.datetime] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    models = payload.get("models")
    fetched_at = _as_datetime(payload.get("fetched_at"))
    if not isinstance(models, list) or fetched_at is None:
        return None
    return models, fetched_at


def _write_cache(cache_path: Path, raw_models: list[dict], fetched_at: dt.datetime) -> None:
    """Atomic, UTF-8 regardless of the platform's default encoding."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": fetched_at.isoformat(),
        "source": CATALOG_URL,
        "models": raw_models,
    }
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=cache_path.parent,
        prefix=cache_path.name,
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(handle.name, cache_path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


async def fetch_catalog(
    *,
    cache_path: Path | None = None,
    client: httpx.AsyncClient | None = None,
    now: dt.datetime | None = None,
    ttl_hours: int = CACHE_TTL_HOURS,
    staleness_months: int = DEFAULT_STALENESS_MONTHS,
    force: bool = False,
) -> Catalog:
    """Return a tiered catalog, preferring a fresh cache over a network call.

    The cache stores the raw registry payload rather than tiered output, so
    changing ``staleness_months`` re-tiers without a refetch.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    cache_path = cache_path or default_cache_path()

    cached = _read_cache(cache_path)
    if cached and not force:
        raw_models, fetched_at = cached
        if fetched_at + dt.timedelta(hours=ttl_hours) > now:
            return build_catalog(
                raw_models, now=fetched_at, staleness_months=staleness_months
            )

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await client.get(CATALOG_URL)
        response.raise_for_status()
        raw_models = response.json().get("data") or []
    except (httpx.HTTPError, ValueError) as error:
        if cached:
            raw_models, fetched_at = cached
            return build_catalog(
                raw_models, now=fetched_at, staleness_months=staleness_months, stale=True
            )
        raise CatalogUnavailable(
            f"cannot reach {CATALOG_URL} and no cache at {cache_path}: {error}"
        ) from error
    finally:
        if owns_client:
            await client.aclose()

    _write_cache(cache_path, raw_models, now)
    return build_catalog(raw_models, now=now, staleness_months=staleness_months)
