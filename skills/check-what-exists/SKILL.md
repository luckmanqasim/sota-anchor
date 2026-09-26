---
name: check-what-exists
description: Use before answering any request that rests on something being unavailable - a closed or undocumented file format, a vendor SDK or tool the user cannot use, a missing API or library - whether the plan is to build around it (a custom parser, reader or converter, a rewrite from scratch, reverse-engineering, an OCR or heuristic stage around a model limitation) or to advise against building it. Run it before telling the user that no open-source tool exists, that only the vendor's software can read a format, or what the realistic options are; those are exactly the claims training data gets wrong, and this checks them against recent repositories, packages and papers. Also use when the user asks whether an approach is still state of the art, or invokes /sota-check.
---

# Checking an assumed limitation

Plans often rest on something being unavailable: no library for the job, no
reader for a format without the vendor's SDK, no API for the data, no model that
can do the task directly. Training data has a cutoff, and any of those may have
changed since - through a new open-source implementation, a package, a tool or a
model capability. Building around a limitation that no longer exists is work that
was obsolete before it was written.

Advice is covered too. Telling the user that nothing open-source reads a format,
or that the vendor's software is the only route, is a claim about what exists
today - the claim this skill checks - not an answer to give from memory.

This skill checks that, and you do the reasoning - there is no second model
involved. The plugin supplies dated, cited evidence; you judge against it.

## The one rule

**Never judge from memory.** Recollection of what exists and what tools can do
is the thing being checked. Judge only from the evidence the tool returns, in
either direction: training data can neither establish that a limitation has
fallen nor that it holds.

## Step 1: invert the proposal

Take the design under review and ask: *what must be unavailable, impossible or
impractical for this specific workaround to be justified?* The answer is often
not about AI at all.

Keep constraints the user states as given. "I can't use ODA" is a constraint - a
license, a platform, a dependency they cannot add - not a claim to overturn. The
assumption to check is the one underneath the workaround: usually that nothing
else already satisfies that constraint.

Write down:

- `domain` - the field this sits in
- `implicit_limitation` - the claim the design depends on, stated so that
  evidence could contradict it
- `proposed_workaround` - the machinery being built to route around it
- `domain_query` - keywords for the specific task, in its own field's vocabulary
- `capability_query` - keywords for whatever would make the workaround
  unnecessary: an existing implementation, library, tool or model capability,
  in vocabulary that field may never use

Lead each query with its most specific term - a file extension, format, product,
library or protocol name - and put generic words last. Retrieval relaxes a query
that finds nothing by dropping its last terms first, so the distinctive term is
the one that should survive.

Two worked examples:

- *"Write a custom binary parser for NWD files, since we can't use ODA to read
  them."* Limitation: no open-source reader exists for NWD without the vendor's
  SDK. Workaround: reverse-engineer the format and write a parser from scratch.
  `domain_query`: "nwd glb conversion". `capability_query`: "nwd reader".
- *"OCR bounding box snapping for pipe penetrations."* Limitation: models cannot
  extract vector coordinates from drawings directly. `domain_query`: "MEP
  penetration extraction construction drawings". `capability_query`:
  "multimodal vector coordinate extraction".

Two queries, because they fail differently. The domain query finds work that
names the task; the capability query finds a general advance - a library, a
reader, a model capability - indexed under vocabulary the task's field never
uses. A single narrow query misses exactly the leaps worth knowing about.

## Step 2: retrieve evidence

**Use the MCP tool. It is the primary path:**

```
check_what_exists(pitch="<the design>",
                  domain_query="<domain_query>",
                  capability_query="<capability_query>")
```

It returns the evidence *and* the standard of proof, already assembled: papers
from arXiv and Hugging Face, repositories from GitHub and packages from npm and
crates.io, each dated. Called with only a pitch it returns the step-1 inversion
prompt instead, so it is safe to start there if you skipped ahead.

Pass `months=24` to widen the window when a first pass returns nothing usable.

**Fallback, only if the MCP server is unreachable.** The CLI does the same
retrieval from the plugin's own checkout. It is a fallback because it has to be
resolved first: a bare `sota-anchor` usually fails with
`exit code 127: command not found`, since a plugin install puts nothing on
`PATH`. Use the first of these that works:

```bash
uv run --quiet --project "${CLAUDE_PLUGIN_ROOT}" sota-anchor --version
"${CLAUDE_PLUGIN_ROOT}/.venv/bin/sota-anchor" --version          # POSIX
"${CLAUDE_PLUGIN_ROOT}/.venv/Scripts/sota-anchor.exe" --version  # Windows
command -v sota-anchor                                          # installed globally
```

The `--project` flag matters: without it uv resolves against the user's project
instead of the plugin, finds no `sota-anchor` there, and creates a `.venv` in it.

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
   documents that a modern primitive, tool, implementation or method has
   superseded it.
3. **Threshold, by kind of claim.**
   - *Existence* - "no library, reader or tool exists for this": a published
     repository or package that states it does the task documents that one
     exists. Report its age and activity too; existence is not maturity.
   - *Performance* - "models cannot do this accurately": prefer benchmarked or
     demonstrated results over work that merely proposes an approach.

   Evidence that is absent, incomplete or off-target meets neither threshold,
   and the assumption stands.

If the evidence documents supersession, report exactly this:

```
[SOTA ARBITER PARADIGM SHIFT]
- Assertion (A): Do NOT implement [the workaround].
- Reason (R): [the native primitive, tool or implementation that supersedes it,
  naming which evidence item says so].
- Linkage: Because (R) is true, (A) is obsolete technical debt.
```

Name the evidence item in the Reason. An assertion the reader cannot trace back
to a paper, repository or package is indistinguishable from a guess.

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
so it works the same on a drawing pipeline, a file-format converter, a genomics
parser or a compiler pass. Do not narrow it to AI model capabilities, or to
fields you recognise: any plan that exists because something is assumed to be
missing is in scope.
