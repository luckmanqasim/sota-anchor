"""Channel 2 - zero-shot assumption inversion and reality differ.

Three stages, none of which knows anything about any discipline:

1. **Invert.** Ask what would have to be hard or impossible for the proposal's
   workaround to be justified. The output is a falsifiable claim plus a search
   query, both produced by the model rather than looked up in a table.
2. **Differ.** Retrieve recent evidence for that query (see :mod:`retriever`).
3. **Judge.** Ask whether the evidence has made the claim obsolete, and if so
   return it as an assertion, its reason, and the link between them.

On the assertion-reason shape: SciUnlearn (Paul, Patwardhan & Cohan,
arXiv:2608.20960) shows that weight-level unlearning of outdated scientific
claims achieves only "superficial suppression", which is the empirical case for
correcting an agent *in context* instead of expecting its weights to have moved
on. Assertion-reason is one of the four QA formats in that benchmark, borrowed
here as the output shape. The belief that phrasing an update as assertion plus
reason makes an agent act on it is this project's design hypothesis, not a
result from that paper.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from .catalog import Catalog, CatalogUnavailable, fetch_catalog
from .retriever import DEFAULT_WINDOW_MONTHS, EvidenceSet, gather_evidence

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
MAX_JSON_ATTEMPTS = 3


class ArbiterError(RuntimeError):
    """The model did not return usable output."""


class LLMUnavailable(ArbiterError):
    """No usable LLM configuration."""


class JSONCompleter(Protocol):
    async def complete_json(self, prompt: str) -> dict[str, Any]: ...


class Inversion(BaseModel):
    domain: str
    implicit_limitation: str
    proposed_workaround: str
    domain_query: str
    capability_query: str

    @field_validator(
        "domain",
        "implicit_limitation",
        "proposed_workaround",
        "domain_query",
        "capability_query",
    )
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    def query_vectors(self) -> list[str]:
        """Both search vectors, in order of specificity.

        Two rather than one because they fail differently: a narrow domain query
        misses a general advance indexed under other terminology, and a broad
        capability query misses work that only names the specific task.
        """
        return [self.domain_query, self.capability_query]


class Verdict(BaseModel):
    """Either an obsolescence finding or a reasoned refusal to make one."""

    is_obsolete: bool
    assertion: str | None = None
    reason: str | None = None
    logical_linkage: str | None = None
    rationale: str | None = None

    @field_validator("is_obsolete", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return value

    def model_post_init(self, _context: Any) -> None:
        if self.is_obsolete:
            missing = [
                name
                for name in ("assertion", "reason", "logical_linkage")
                if not (getattr(self, name) or "").strip()
            ]
            if missing:
                raise ValueError(f"obsolete verdict missing {', '.join(missing)}")
        elif not (self.rationale or "").strip():
            raise ValueError("non-obsolete verdict requires a rationale")


class LLMSettings(BaseModel):
    api_key: str
    base_url: str
    model: str | None = None


def resolve_settings(env: Mapping[str, str] | None = None) -> LLMSettings:
    """Resolve LLM configuration from the environment.

    OpenRouter is preferred when both keys are present: the catalog already comes
    from OpenRouter, so one key covers both channels.
    """
    env = env if env is not None else os.environ

    explicit = (env.get("SOTA_ANCHOR_API_KEY") or "").strip()
    if explicit:
        return LLMSettings(
            api_key=explicit,
            base_url=(env.get("SOTA_ANCHOR_BASE_URL") or OPENROUTER_BASE_URL).strip(),
            model=(env.get("SOTA_ANCHOR_MODEL") or "").strip() or None,
        )

    for key_name, base_url in (
        ("OPENROUTER_API_KEY", OPENROUTER_BASE_URL),
        ("OPENAI_API_KEY", OPENAI_BASE_URL),
    ):
        key = (env.get(key_name) or "").strip()
        if key:
            return LLMSettings(
                api_key=key,
                base_url=(env.get("SOTA_ANCHOR_BASE_URL") or base_url).strip(),
                model=(env.get("SOTA_ANCHOR_MODEL") or "").strip() or None,
            )

    raise LLMUnavailable(
        "no API key found: set SOTA_ANCHOR_API_KEY (or OPENROUTER_API_KEY / OPENAI_API_KEY)"
    )


CatalogFetcher = Callable[..., Awaitable[Catalog]]


async def resolve_model(
    settings: LLMSettings,
    *,
    fetch: CatalogFetcher | None = None,
    **fetch_kwargs: Any,
) -> str:
    """Return the judge model, resolving an unset one from the live catalog.

    Not hardcoding a judge model is the same discipline the tool asks of its
    users: the model that arbitrates obsolescence should not itself be a stale
    constant.
    """
    if settings.model:
        return settings.model

    fetch = fetch or fetch_catalog
    try:
        catalog = await fetch(**fetch_kwargs)
    except CatalogUnavailable as error:
        raise LLMUnavailable(
            f"no model configured and the catalog is unreachable: {error}. "
            "Set SOTA_ANCHOR_MODEL to choose one explicitly."
        ) from error

    for candidate in catalog.featured():
        return candidate.id
    for candidate in catalog.featured(providers=None):
        return candidate.id
    raise LLMUnavailable(
        "no model configured and the catalog offered no usable endpoint; set SOTA_ANCHOR_MODEL"
    )


RawCompleter = Callable[[str], Awaitable[str]]


class LLMClient:
    """An OpenAI-compatible JSON completer with tolerant parsing.

    ``completion_fn`` is injectable so the parsing and retry behaviour can be
    exercised without a network or a provider account.
    """

    def __init__(self, settings: LLMSettings, *, completion_fn: RawCompleter | None = None):
        self.settings = settings
        self._completion_fn = completion_fn or self._openai_completion
        self._client: Any = None

    async def ensure_model(self) -> str:
        """Resolve the judge model on first use.

        Kept inside the client so callers only ever need a ``JSONCompleter``:
        having them reach into ``settings.model`` made the model a caller
        responsibility and every test double had to grow a settings object.
        """
        if not self.settings.model:
            self.settings.model = await resolve_model(self.settings)
        return self.settings.model

    async def _openai_completion(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        model = await self.ensure_model()
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.api_key, base_url=self.settings.base_url
            )
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(MAX_JSON_ATTEMPTS):
            raw = await self._completion_fn(prompt)
            try:
                return _parse_json_object(raw)
            except ValueError as error:
                last_error = error
        raise ArbiterError(
            f"model did not return a JSON object: {last_error}"
        ) from last_error


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fences and surrounding prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except ValueError as error:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in output") from error
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


INVERSION_PROMPT = """Given the following technical task or architecture proposal:
"{proposal}"

Analyze the proposal and identify the assumption it takes for granted. Answer
the question: "What must be unavailable, impossible or impractical for this
specific workaround/design to be justified?" The answer is often not about AI:
no library or reader for a file format, only a vendor SDK able to open it, no
API for the data, no model able to do the task directly.

Keep constraints the proposal states as given - a license that cannot be used, a
dependency that cannot be added. The assumption to check is the one underneath
the workaround: usually that nothing else already satisfies those constraints.

Output valid JSON matching this schema:
{{
  "domain": string,
  "implicit_limitation": string,
  "proposed_workaround": string,
  "domain_query": string,
  "capability_query": string
}}

Both queries must be short keyword search phrases, not questions, suitable for
searching recent research, repositories and package registries. Lead each with
its most specific term - a file extension, format, product, library or protocol
name - and put generic words last.

- domain_query names the specific task in this domain's own vocabulary.
- capability_query names whatever would make the workaround unnecessary - an
  existing implementation, library, tool or model capability - in vocabulary
  the task's own field may not use.

Two queries, not one, because they fail differently. A narrow domain query
misses a general advance indexed under other terminology; a broad capability
query misses work that only ever names the specific task."""


JUDGE_PROMPT = """You are an Epistemic Conflict Arbiter evaluating technical obsolescence.

Proposed Assumption: "{implicit_limitation}"
Proposed Workaround: "{proposed_workaround}"
Recent SOTA Evidence (past {months} months):
"{evidence}"

Question: Has recent tooling, a published implementation or package, frontier
model capabilities, or open-source advances in the past {months} months rendered
this limitation/workaround obsolete?

Standard of proof:

1. Evidence-only invariant. Judge from the evidence above and nothing else.
   Training data is not admissible in either direction: it can neither
   establish that the limitation has fallen nor that it holds.
2. Default baseline. The assumption stands unless the evidence explicitly
   documents that a modern primitive, tool, implementation or method has
   superseded it.
3. Threshold, which depends on the kind of claim:
   - Existence ("no library, reader or tool exists for this"): a published
     repository or package whose description states that it does the task
     documents that one exists. Mention its age and activity, because
     existence is not maturity.
   - Performance ("models cannot do this accurately"): prefer evidence
     reporting benchmarked or demonstrated results over evidence that merely
     proposes an approach.
   Evidence that is absent, incomplete or off-target meets neither threshold,
   and neither does an unbenchmarked result offered for a performance claim.
   In those cases the assumption stands by rule 2.

If YES, return valid JSON:
{{
  "is_obsolete": true,
  "assertion": "Do NOT build [proposed_workaround].",
  "reason": "[Explain what modern native primitive or tool replaces it, based on the evidence].",
  "logical_linkage": "Because [reason], [assertion] represents redundant technical debt."
}}

If NO (the limitation is still genuine and requires engineering workarounds), return:
{{
  "is_obsolete": false,
  "rationale": "[Brief explanation why this is still an active limitation]"
}}"""


async def invert(proposal: str, *, llm: JSONCompleter) -> Inversion:
    """Stage 1: surface the limitation the proposal takes for granted."""
    if not proposal.strip():
        raise ArbiterError("nothing to analyse: the proposal is empty")

    prompt = INVERSION_PROMPT.format(proposal=proposal.strip())
    last_error: Exception | None = None
    for _ in range(MAX_JSON_ATTEMPTS):
        payload = await llm.complete_json(prompt)
        try:
            return Inversion.model_validate(payload)
        except ValidationError as error:
            last_error = error
    raise ArbiterError(f"could not invert the proposal: {last_error}")


def _no_evidence_verdict(evidence: EvidenceSet) -> Verdict:
    rationale = (
        "No recent evidence was retrieved, so no obsolescence claim can be made. "
        "This is not a finding that the limitation still holds."
    )
    if evidence.errors:
        rationale += " Retrieval errors: " + "; ".join(evidence.errors)
    return Verdict(is_obsolete=False, rationale=rationale)


async def judge(inversion: Inversion, evidence: EvidenceSet, *, llm: JSONCompleter) -> Verdict:
    """Stage 3: decide whether the evidence retires the limitation.

    With no evidence the model is never asked. A judge handed an empty evidence
    block can still produce a confident YES from its own priors, and telling a
    developer to abandon work on the strength of nothing is the worst thing this
    tool could do.
    """
    if evidence.is_empty:
        return _no_evidence_verdict(evidence)

    prompt = JUDGE_PROMPT.format(
        implicit_limitation=inversion.implicit_limitation,
        proposed_workaround=inversion.proposed_workaround,
        months=evidence.window_months,
        evidence=evidence.render(),
    )
    last_error: Exception | None = None
    for _ in range(MAX_JSON_ATTEMPTS):
        payload = await llm.complete_json(prompt)
        try:
            return Verdict.model_validate(payload)
        except ValidationError as error:
            last_error = error
    raise ArbiterError(f"could not read the arbiter verdict: {last_error}")


def _asciify(text: str) -> str:
    import unicodedata

    # Escapes rather than literals: this table exists to match these exact
    # code points, and a literal smart quote in source is easy to mangle.
    folded = (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")


def render_paradigm_update(verdict: Verdict) -> str:
    """The system-prompt injection block for an obsolete assumption."""
    if not verdict.is_obsolete:
        raise ValueError("a paradigm update is only meaningful for an obsolete verdict")
    return "\n".join(
        [
            "[SOTA ARBITER PARADIGM UPDATE]",
            f"- Assertion (A): {_asciify(verdict.assertion or '')}",
            f"- Reason (R): {_asciify(verdict.reason or '')}",
            f"- Linkage: {_asciify(verdict.logical_linkage or '')}",
        ]
    )


class Report(BaseModel):
    proposal: str
    inversion: Inversion
    evidence: EvidenceSet
    verdict: Verdict
    checked_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def render(self) -> str:
        lines: list[str] = []
        if self.verdict.is_obsolete:
            lines += [render_paradigm_update(self.verdict), ""]
        else:
            lines += [
                "[SOTA ARBITER: NO PARADIGM SHIFT]",
                f"- Domain: {_asciify(self.inversion.domain)}",
                f"- Assumption holds: {_asciify(self.inversion.implicit_limitation)}",
                f"- Rationale: {_asciify(self.verdict.rationale or '')}",
                "",
            ]

        lines += [f"Evidence considered (past {self.evidence.window_months} months):"]
        if self.evidence.is_empty:
            lines.append("  none retrieved")
        for item in self.evidence.items:
            lines.append(
                f"  [{item.source}] {item.published.isoformat()} "
                f"{_asciify(item.title)} - {item.url}"
            )
        if self.evidence.errors:
            lines += ["", "Retrieval errors:"]
            lines += [f"  {_asciify(error)}" for error in self.evidence.errors]
        return "\n".join(lines).rstrip() + "\n"


EvidenceGatherer = Callable[..., Awaitable[EvidenceSet]]


async def check_what_exists(
    proposal: str,
    *,
    llm: JSONCompleter,
    gather: EvidenceGatherer | None = None,
    now: dt.datetime | None = None,
    months: int = DEFAULT_WINDOW_MONTHS,
    **gather_kwargs: Any,
) -> Report:
    """Run the full three-stage pipeline over an arbitrary technical proposal."""
    gather = gather or gather_evidence
    now = now or dt.datetime.now(dt.UTC)

    inversion = await invert(proposal, llm=llm)
    evidence = await gather(
        inversion.query_vectors(), now=now, months=months, **gather_kwargs
    )
    verdict = await judge(inversion, evidence, llm=llm)
    return Report(
        proposal=proposal,
        inversion=inversion,
        evidence=evidence,
        verdict=verdict,
        checked_at=now,
    )
