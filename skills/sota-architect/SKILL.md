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
tool returns. If no evidence comes back, there is no verdict - see step 4.

## Step 1: invert the proposal

Take the design under review and ask: *what must be hard, impossible, or
inaccurate for AI or software in this domain for this specific workaround to be
justified?*

Write down four things:

- `domain` - the field this sits in
- `implicit_limitation` - the capability claim the design depends on being true
- `proposed_workaround` - the machinery being built to route around it
- `verification_query` - a short keyword search phrase, not a question, that
  would surface recent work showing the limitation has fallen

The query matters more than it looks. "OCR bounding box snapping for pipe
penetrations" describes the workaround; "direct vector polygon extraction
technical drawings" describes the capability that would make it unnecessary.
Search for the capability, not the workaround.

## Step 2: retrieve evidence

Run the retrieval, substituting your query:

```bash
sota-anchor evidence --query "<your verification_query>" --json
```

Use `--months 6` to tighten the window, `--months 24` to widen it when a first
pass returns nothing. If the MCP server is configured you can call
`verify_architecture` instead, which returns the same evidence with the judging
protocol attached.

Retrieval hits arXiv and GitHub with a rolling recency window. It reports its
own failures: if you see a `throttled` error, that source contributed nothing and
the evidence set is thinner than it looks. Read the errors before you read the
results.

## Step 3: judge against the evidence

If the evidence shows the limitation has been overcome, report exactly this:

```
[SOTA ARBITER PARADIGM SHIFT]
- Assertion (A): Do NOT implement [the workaround].
- Reason (R): [the native primitive or tool that replaces it, naming which
  evidence item says so].
- Linkage: Because (R) is true, (A) is obsolete technical debt.
```

Name the evidence item in the Reason. An assertion the user cannot trace back to
a paper or repository is indistinguishable from a guess.

If the evidence does not show that, say so plainly instead: the limitation still
appears genuine, here is what was searched, and here is what would have to be
true to change the answer. Do not emit the block in that case. A protocol that
only describes how to say yes is a protocol for saying yes.

## Step 4: when nothing comes back

An empty evidence set is **not** a finding that the limitation holds, and **not**
a finding that it has fallen. Nothing was checked.

Say that plainly, proceed with the original plan, and offer to widen the window
or search manually. Do not produce a paradigm-shift block - with no evidence in
front of you, any conclusion would come from training data, which is exactly
what this check exists to distrust.

## Scope

The engine is domain-agnostic: it holds no list of topics, tools or disciplines,
so it works the same on a technical-drawing pipeline, a genomics parser or a
compiler pass. Do not narrow it to fields you recognise.
