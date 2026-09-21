---
name: sota-architect
description: Use before building any workaround for something a tool "cannot" do - an OCR or parsing stage, a heuristic pipeline, a custom post-processing step, a hand-rolled scheduler. Checks whether the limitation the design routes around still exists, using research and repositories from the last 6-12 months rather than training data. Also use when the user asks whether an approach is still state of the art, or invokes /sota-check.
---

# Architectural obsolescence arbitration

Your training data has a cutoff. A limitation you remember as real may have been
removed by tooling, frontier model capability, or open-source work published
since. Every hour spent building around a limitation that no longer exists is
technical debt that was obsolete before it was written.

This skill checks that, and you do the reasoning - there is no second model
involved. The plugin supplies dated, cited evidence; you judge against it.

## The one rule

**Never judge obsolescence from memory.** Your recollection of what models and
tools can do is the thing under suspicion here. Judge only from the evidence the
tool returns, in either direction: training data can neither establish that a
limitation has fallen nor that it holds.

## Step 1: invert the proposal

Take the design under review and ask: *what must be hard, impossible, or
inaccurate for AI or software in this domain for this specific workaround to be
justified?*

Write down:

- `domain` - the field this sits in
- `implicit_limitation` - the capability claim the design depends on being true
- `proposed_workaround` - the machinery being built to route around it
- `domain_query` - keywords for the specific task, in its own field's vocabulary
- `capability_query` - keywords for the broad, foundational capability that would
  make the workaround unnecessary, in vocabulary that field may never use

Two queries, because they fail differently. "OCR bounding box snapping for pipe
penetrations" describes the workaround. "MEP penetration extraction construction
drawings" is the domain query. "Multimodal direct vector coordinate extraction"
is the capability query - and it is the one that finds a foundational advance
published without ever naming your domain. A single narrow query misses exactly
the leaps worth knowing about.

## Step 2: retrieve evidence

**Use the MCP tool. It is the primary path:**

```
verify_architecture(pitch="<the design>",
                    domain_query="<domain_query>",
                    capability_query="<capability_query>")
```

It returns the evidence *and* the standard of proof, already assembled. Called
with only a pitch it returns the step-1 inversion prompt instead, so it is safe
to start there if you skipped ahead.

Pass `months=24` to widen the window when a first pass returns nothing usable.

**Fallback, only if the MCP server is unreachable.** The CLI does the same
retrieval. It is a fallback because it depends on `sota-anchor` being resolvable,
which it often is not - the package commonly lives in a project `.venv` rather
than on the system `PATH`, and calling it blind produces
`exit code 127: command not found`. Resolve it before calling it:

```bash
# Use the first of these that exists; do not just call `sota-anchor`.
./.venv/bin/sota-anchor --version          # POSIX project venv
./.venv/Scripts/sota-anchor.exe --version  # Windows project venv
uv run --quiet sota-anchor --version       # uv-managed project
command -v sota-anchor                     # already on PATH
```

Then run retrieval with whichever resolved:

```bash
<resolved> evidence --query "<domain_query>" --json
<resolved> evidence --query "<capability_query>" --json
```

If none resolves, say so plainly and stop - do not guess a verdict. A retrieval
you could not run is not evidence that the limitation holds.

Retrieval reports its own failures. A `throttled` error means that source
contributed nothing and the evidence set is thinner than it looks. Read the
errors before you read the results.

## Step 3: judge against the evidence

Apply the standard of proof the tool returns:

1. **Evidence-only.** Judge from the retrieved evidence and nothing else.
2. **Default baseline.** The assumption stands unless the evidence explicitly
   documents that a modern primitive, tool or method has superseded it.
3. **Threshold.** Prefer evidence reporting benchmarked or demonstrated results
   over evidence that merely proposes an approach. Unbenchmarked or incomplete
   evidence does not meet the threshold, and the assumption stands.

If the evidence documents supersession, report exactly this:

```
[SOTA ARBITER PARADIGM SHIFT]
- Assertion (A): Do NOT implement [the workaround].
- Reason (R): [the native primitive or tool that supersedes it, naming which
  evidence item says so].
- Linkage: Because (R) is true, (A) is obsolete technical debt.
```

Name the evidence item in the Reason. An assertion the reader cannot trace back
to a paper or repository is indistinguishable from a guess.

Otherwise state that the assumption stands on this evidence, say what the
evidence does and does not document, and name what would have to be shown to
overturn it. Do not emit the block in that case. A protocol that only describes
how to say yes is a protocol for saying yes.

## Step 4: when nothing comes back

An empty evidence set is **not** a finding that the limitation holds, and **not**
a finding that it has fallen. Nothing was checked, so nothing was learned.

Say that plainly. The assumption stands by default, but be precise about why:
not because the retrieved record documents that it holds, but because there is
no retrieved record. Do not state or imply that recent work has been surveyed.
Offer to widen the window or search manually, and proceed with the original plan.

## Scope

The engine is domain-agnostic: it holds no list of topics, tools or disciplines,
so it works the same on a technical-drawing pipeline, a genomics parser or a
compiler pass. Do not narrow it to fields you recognise.
