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
* the protocol says in plain words that memory is not admissible;
* and when retrieval comes back empty, no judging template is emitted at all.

The module is pure. No network, no filesystem, no LLM client.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .retriever import DEFAULT_WINDOW_MONTHS, EvidenceSet, gather_evidence

PHASE_INVERT = "invert"
PHASE_JUDGE = "judge"
PHASE_NO_EVIDENCE = "no_evidence"

Phase = Literal["invert", "judge", "no_evidence"]

PARADIGM_HEADER = "[SOTA ARBITER PARADIGM SHIFT]"


def _asciify(text: str) -> str:
    """Flatten to ASCII; these payloads cross hook stdout and cp1252 consoles."""
    folded = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")


@dataclass(frozen=True)
class Payload:
    """One instruction block for the host agent, plus what produced it."""

    phase: Phase
    body: str
    pitch: str
    verification_query: str | None = None
    evidence: EvidenceSet | None = None

    def render(self) -> str:
        return self.body


INVERSION_TEMPLATE = """[SOTA ARBITER - STEP 1 OF 2: ASSUMPTION INVERSION]

You are about to verify whether a technical proposal rests on a limitation that
no longer exists. Do not answer that question yet - you have no evidence, and
your own training data is the thing under suspicion.

Proposal under review:
"{pitch}"

First, invert it. Identify the engineering assumption the proposal takes for
granted by answering: what must be hard, impossible, or inaccurate for AI or
software in this domain for this specific workaround or design to be justified?

Produce these four fields:

- domain: the field this sits in
- implicit_limitation: the capability claim the design depends on being true
- proposed_workaround: the machinery being built to route around it
- verification_query: a short keyword search phrase - not a question - that
  would surface recent research or code showing the limitation has fallen

Then call this tool again with the same pitch and your verification_query. Step
2 will return real, dated evidence and the protocol for judging against it.
"""


JUDGMENT_TEMPLATE = """[SOTA ARBITER - STEP 2 OF 2: EPISTEMIC CONFLICT JUDGMENT]

Proposal under review:
"{pitch}"

Verification query used: {query}

{evidence}

Judge strictly from the evidence above. Your training data has a cutoff and is
not admissible here: if the evidence does not itself demonstrate that the
limitation has fallen, the answer is no. An unsupported "yes" tells a developer
to abandon work they still need.

If the evidence shows the limitation has been overcome, reply with exactly this
block, filled in:

{header}
- Assertion (A): Do NOT implement [the workaround].
- Reason (R): [the modern native primitive or tool that replaces it, naming the
  evidence item it comes from].
- Linkage: Because (R) is true, (A) is obsolete technical debt.

If the evidence does not show that, say so plainly instead: state that the
limitation still appears genuine, and say what would have to be true to change
that. Do not emit the block in that case.
"""


NO_EVIDENCE_TEMPLATE = """[SOTA ARBITER - NO VERDICT POSSIBLE]

Proposal under review:
"{pitch}"

Verification query used: {query}

No recent evidence was retrieved for this query.{errors}

This is NOT a finding that the limitation still holds, and it is NOT a finding
that it has fallen. Nothing was checked. Do not conclude that anything is
obsolete, and do not produce a paradigm-shift block - with no evidence in front
of you, any such conclusion would come from training data, which is the thing
this check exists to distrust.

Proceed with the original plan. If the question matters, say that verification
returned nothing and offer to retry, widen the window with --months, or search
manually.
"""


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


def build_judgment_request(pitch: str, verification_query: str, evidence: EvidenceSet) -> Payload:
    """Step 2: hand over the evidence and the protocol - or refuse.

    With an empty evidence set this returns the refusal directive rather than a
    judging template. The distinction is the whole guard: a model given an
    Assertion-Reason form and nothing to fill it from will fill it from priors.
    """
    cleaned = pitch.strip()
    query = verification_query.strip()

    if evidence.is_empty:
        return Payload(
            phase=PHASE_NO_EVIDENCE,
            body=_asciify(
                NO_EVIDENCE_TEMPLATE.format(
                    pitch=cleaned, query=query or "(none)", errors=_render_errors(evidence)
                )
            ),
            pitch=cleaned,
            verification_query=query or None,
            evidence=evidence,
        )

    block = evidence.render().rstrip()
    if evidence.errors:
        block += _render_errors(evidence)

    return Payload(
        phase=PHASE_JUDGE,
        body=_asciify(
            JUDGMENT_TEMPLATE.format(
                pitch=cleaned, query=query, evidence=block, header=PARADIGM_HEADER
            )
        ),
        pitch=cleaned,
        verification_query=query,
        evidence=evidence,
    )


EvidenceGatherer = Callable[..., Awaitable[EvidenceSet]]


async def build_verification_payload(
    pitch: str,
    *,
    verification_query: str | None = None,
    gather: EvidenceGatherer | None = None,
    months: int = DEFAULT_WINDOW_MONTHS,
    **gather_kwargs: Any,
) -> Payload:
    """Dispatch the two-phase protocol.

    No query means the inversion has not happened yet, so nothing is retrieved -
    searching on the raw pitch would look busy while querying the wrong thing.
    """
    if not verification_query or not verification_query.strip():
        return build_inversion_request(pitch)

    gather = gather or gather_evidence
    evidence = await gather(verification_query.strip(), months=months, **gather_kwargs)
    return build_judgment_request(pitch, verification_query, evidence)
