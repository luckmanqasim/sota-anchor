"""Host-driven verification payloads - the zero-key reasoning path.

Instead of spending a second LLM account on inversion and judgment, this module
builds the prompts the *host* agent answers. The plugin keeps the part that
needs no intelligence (deterministic, dated, cited retrieval) and hands back the
part that does.

That inverts who is trusted. The host model is the one carrying the stale priors
this tool exists to correct, so asking it "is this obsolete?" is circular unless
it is answering from documents rather than memory. Everything here is built
around that constraint:

* evidence comes first in the payload, dated and cited;
* the burden of proof is stated objectively, with model priors inadmissible;
* frontier grounding, when supplied, is a dated registry snapshot rather than an
  assertion about what a named model can do;
* and when retrieval comes back empty, no judging template is emitted at all.

The module is pure. No network, no filesystem, no LLM client.
"""

from __future__ import annotations

import textwrap
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .catalog import Catalog
from .retriever import DEFAULT_WINDOW_MONTHS, EvidenceSet, gather_evidence

PHASE_INVERT = "invert"
PHASE_JUDGE = "judge"
PHASE_NO_EVIDENCE = "no_evidence"

Phase = Literal["invert", "judge", "no_evidence"]

PARADIGM_HEADER = "[SOTA ARBITER PARADIGM SHIFT]"


def _asciify(text: str) -> str:
    """Flatten to ASCII; these payloads cross hook stdout and cp1252 consoles."""
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


@dataclass(frozen=True)
class Payload:
    """One instruction block for the host agent, plus what produced it."""

    phase: Phase
    body: str
    pitch: str
    queries: tuple[str, ...] = ()
    evidence: EvidenceSet | None = None

    def render(self) -> str:
        return self.body


INVERSION_TEMPLATE = """[SOTA ARBITER - STEP 1 OF 2: ASSUMPTION INVERSION]

This step identifies what the proposal assumes, so that step 2 can check it
against retrieved, dated sources. No verdict is asked for yet: nothing has been
retrieved, and recall from training data is what the check is designed not to
rely on.

Proposal under review:
"{pitch}"

Invert it. What must be unavailable, impossible or impractical for this specific
workaround or design to be justified? The answer is often not about AI at all:
no library or reader exists for a file format, only a vendor SDK can open it, an
API does not expose the data, no model can do the task directly.

Keep constraints the proposal states as given - a license that cannot be used, a
dependency that cannot be added. The assumption to check is the one underneath
the workaround: usually that nothing else already satisfies those constraints.

Produce these fields:

- domain: the field this sits in
- implicit_limitation: the claim the design depends on being true, stated so
  that evidence could contradict it
- proposed_workaround: the machinery being built to route around it
- domain_query: keywords for the specific task, in this domain's own vocabulary
- capability_query: keywords for whatever would make the workaround unnecessary
  - an existing implementation, library, tool or model capability - in
  vocabulary the task's own field may not use

Lead each query with its most specific term - a file extension, format, product,
library or protocol name - and put generic words last. Retrieval relaxes a query
that finds nothing by dropping its last terms first.

Two queries, not one, because they fail differently. A narrow domain query
misses a general advance indexed under other terminology; a broad capability
query misses work that only ever names the specific task. Searching both is what
catches a foundational leap that never mentions your domain by name.

Then call this tool again with the same pitch and both queries. Step 2 will
return real, dated evidence and the standard for judging against it.
"""


JUDGMENT_TEMPLATE = """[SOTA ARBITER - STEP 2 OF 2: EPISTEMIC CONFLICT JUDGMENT]

Proposal under review:
"{pitch}"

Queries used: {queries}
{snapshot}
{evidence}

Standard of proof:

1. Evidence-only invariant. Judge from the evidence above and nothing else.
   Recall from training data is not admissible here, in either direction: it
   can neither establish that the limitation has fallen nor that it holds.
2. Default baseline. The assumption stands unless the evidence above explicitly
   documents that a modern primitive, tool, implementation or method has
   superseded it.
3. Threshold, which depends on the kind of claim:
   - Existence ("no library, reader or tool exists for this"): a published
     repository or package whose description states that it does the task
     documents that one exists. Report its age and activity as well, because
     existence is not maturity.
   - Performance ("models cannot do this accurately"): prefer evidence that
     reports benchmarked or demonstrated results over evidence that merely
     proposes or describes an approach.
   Evidence that is absent, incomplete or off-target meets neither threshold,
   and neither does an unbenchmarked result offered for a performance claim.
   In those cases the assumption stands by rule 2.

If the evidence documents supersession, reply with exactly this block, filled in:

{header}
- Assertion (A): Do NOT implement [the workaround].
- Reason (R): [the modern native primitive, tool or implementation that
  supersedes it, naming the evidence item it comes from].
- Linkage: Because (R) is true, (A) is obsolete technical debt.

Otherwise state that the assumption stands on this evidence, say what the
evidence does and does not document, and name what would have to be shown to
overturn it. Do not emit the block in that case.
"""


NO_EVIDENCE_TEMPLATE = """[SOTA ARBITER - NO VERDICT POSSIBLE]

Proposal under review:
"{pitch}"

Queries used: {queries}

No recent evidence was retrieved.{errors}

Nothing was checked, so nothing was learned. By the default baseline the
assumption stands, but note precisely why: not because the retrieved record
documents that it holds, but because there is no retrieved record. Do not state
or imply that recent work has been surveyed.

Do not produce a paradigm-shift block. With no evidence present, any such
conclusion would come from training data, which is inadmissible here.

Proceed with the original plan. If the question matters, say that retrieval
returned nothing and offer to retry, widen the window with a larger months
value, or search manually.
"""


def render_capability_snapshot(catalog: Catalog) -> str:
    """A dated, attributed statement of what current endpoints *declare*.

    This exists so the judge can tell whether a limitation is premised on
    something the registry already contradicts - "models only emit prose", say.

    It reports declared interfaces, never asserted skill. Naming models and
    claiming what they can do would be a static capability dictionary keyed to
    model names: it would go stale, which is the failure this tool exists to
    prevent, and an unsourced capability claim is exactly the kind of prior the
    same prompt rules inadmissible. A dated registry reading is a fact with a
    source, and it updates itself.
    """
    featured = catalog.featured()
    if not featured:
        return ""

    intro = (
        f"Registry snapshot, retrieved {catalog.fetched_at.date().isoformat()} from "
        f"{catalog.source}: the model API endpoints served there and the interfaces "
        "each declares. It lists what can be called over the API; it says nothing "
        "about which model is running this session."
    )
    lines = [*textwrap.wrap(intro, width=76), ""]
    for model in featured:
        accepts = "+".join(model.input_modalities) or "unreported"
        emits = "+".join(model.output_modalities) or "unreported"
        extras = []
        if model.supports_tools:
            extras.append("tool calls")
        if model.supports_structured_output:
            extras.append("structured output")
        suffix = f"; {', '.join(extras)}" if extras else ""
        lines.append(f"  {model.id} - accepts {accepts}, emits {emits}{suffix}")

    lines += [
        "",
        "That is an interface declaration, not a performance claim. It can show",
        "that an assumption about what a model can accept or emit is already out",
        "of date; it cannot show how well any of them performs a task. Only the",
        "evidence below speaks to that.",
    ]
    return "\n".join(lines)


def build_inversion_request(pitch: str) -> Payload:
    """Step 1: ask the host to surface the assumption, with no verdict invited."""
    cleaned = pitch.strip()
    if not cleaned:
        raise ValueError("nothing to verify: the proposal is empty")
    return Payload(
        phase=PHASE_INVERT,
        body=_asciify(INVERSION_TEMPLATE.format(pitch=cleaned)),
        pitch=cleaned,
    )


def _render_errors(evidence: EvidenceSet) -> str:
    if not evidence.errors:
        return ""
    joined = "; ".join(evidence.errors)
    return f"\n\nRetrieval did not complete cleanly: {joined}"


def _normalise_queries(queries: str | Sequence[str] | None) -> tuple[str, ...]:
    if queries is None:
        return ()
    if isinstance(queries, str):
        queries = [queries]
    return tuple(q.strip() for q in queries if q and q.strip())


def build_judgment_request(
    pitch: str,
    queries: str | Sequence[str],
    evidence: EvidenceSet,
    *,
    catalog: Catalog | None = None,
) -> Payload:
    """Step 2: hand over the evidence and the standard - or refuse.

    With an empty evidence set this returns the refusal directive rather than a
    judging template. The distinction is the whole guard: a model given an
    Assertion-Reason form and nothing to fill it from will fill it from priors.
    """
    cleaned = pitch.strip()
    used = _normalise_queries(queries)
    shown = ", ".join(used) if used else "(none)"

    if evidence.is_empty:
        return Payload(
            phase=PHASE_NO_EVIDENCE,
            body=_asciify(
                NO_EVIDENCE_TEMPLATE.format(
                    pitch=cleaned, queries=shown, errors=_render_errors(evidence)
                )
            ),
            pitch=cleaned,
            queries=used,
            evidence=evidence,
        )

    block = evidence.render().rstrip()
    if evidence.errors:
        block += _render_errors(evidence)

    snapshot = render_capability_snapshot(catalog) if catalog is not None else ""
    snapshot_block = f"\n{snapshot}\n" if snapshot else ""

    return Payload(
        phase=PHASE_JUDGE,
        body=_asciify(
            JUDGMENT_TEMPLATE.format(
                pitch=cleaned,
                queries=shown,
                snapshot=snapshot_block,
                evidence=block,
                header=PARADIGM_HEADER,
            )
        ),
        pitch=cleaned,
        queries=used,
        evidence=evidence,
    )


EvidenceGatherer = Callable[..., Awaitable[EvidenceSet]]


async def build_verification_payload(
    pitch: str,
    *,
    domain_query: str | None = None,
    capability_query: str | None = None,
    gather: EvidenceGatherer | None = None,
    catalog: Catalog | None = None,
    months: int = DEFAULT_WINDOW_MONTHS,
    **gather_kwargs: Any,
) -> Payload:
    """Dispatch the two-phase protocol.

    No query means the inversion has not happened yet, so nothing is retrieved -
    searching on the raw pitch would look busy while querying the wrong thing.
    Either vector alone is enough to proceed; both is better.
    """
    vectors = _normalise_queries([domain_query, capability_query])
    if not vectors:
        return build_inversion_request(pitch)

    gather = gather or gather_evidence
    evidence = await gather(list(vectors), months=months, **gather_kwargs)
    return build_judgment_request(pitch, vectors, evidence, catalog=catalog)
