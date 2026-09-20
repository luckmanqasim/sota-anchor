# sota-anchor

Coding agents carry two kinds of stale knowledge. One is trivia: they write
`gpt-4o` into your `.env` because that was current when they were trained. The
other is expensive: they build a workaround for a limitation that no longer
exists — an OCR-and-snapping pipeline, a regex parser, a hand-rolled retry
scheduler — because as far as their weights are concerned, the thing they are
working around is still hard.

`sota-anchor` attacks both, and it does it **without a list of topics**. There is
no keyword table, no domain dictionary, and no hardcoded model roster anywhere in
this codebase. Ask it about MEP pipe penetrations, ribosome profiling, or
compiler IRs and the same code path runs.

---

## The two channels

**Channel 1 — entity anchoring.** Fetches the live model registry and writes a
managed block into `CLAUDE.md`, `.cursorrules`, `.cursor/rules/sota.mdc` or
`AGENTS.md` naming the current endpoints and mapping the superseded ones
forward:

```
| Retired endpoint            | Use instead                 |
| --- | --- |
| `anthropic/claude-opus-4.8` | `anthropic/claude-opus-5`    |
| `google/gemini-3.7-flash`   | `google/gemini-3.8-flash`    |
| `openai/gpt-5.4-pro`        | `openai/gpt-5.5-pro`         |
```

Tiering is derived from the *structure* of a model ID plus registry timestamps,
never from a name list. A slug token containing a digit is a version token, a
purely alphabetic token belongs to the lineage, so `gpt-4o` and `gpt-5.5` share
the `(gpt,)` lineage and supersession resolves between them across a
naming-scheme change. A model is legacy when something newer shares its lineage,
when it trails its *own* provider's newest release by more than the staleness
window, or when the registry has expired it.

**Channel 2 — capability anchoring.** A three-stage pipeline:

1. **Invert.** Given a proposal, ask an LLM what would have to be hard,
   impossible or inaccurate for that design to be justified. It returns a
   falsifiable claim plus a search query — both produced by the model, not looked
   up.
2. **Differ.** Query arXiv and GitHub for work from the past 6–12 months on that
   query.
3. **Judge.** Ask whether the evidence has retired the claim. If it has, the
   answer comes back as an assertion, its reason, and the link between them:

```
[SOTA ARBITER PARADIGM UPDATE]
- Assertion (A): Do NOT build OCR bounding boxes and geometric snapping heuristics.
- Reason (R): Visual-first multimodal RAG systems now return semantic polygon
  annotations from civil standard plans directly.
- Linkage: Because visual-first multimodal RAG returns polygons directly,
  building an OCR snapping pipeline represents redundant technical debt.
```

---

## Install

```bash
uv tool install sota-anchor      # or: uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## Use

```bash
# Refresh instruction files from the live registry. Needs no API key.
sota-anchor sync --target all

# Test a proposal. Exits 0 if the limitation still holds, 2 if it is obsolete.
sota-anchor check "Extract MEP pipe penetrations from PDFs with OCR bounding
                   boxes and geometric snapping heuristics."

# Run as an MCP server over stdio.
sota-anchor serve
```

`sync` options: `--staleness-months` (default 12) for how far behind its own
provider a model may fall before counting as legacy — raise it for providers
that ship slowly; `--provider` (repeatable) and `--all-providers` to widen the
block; `--max-per-provider` (default 4); `--refresh` to bypass the 24-hour cache.

Because `check` exits 2 on an obsolete proposal, it works as a CI gate:

```yaml
- run: sota-anchor check "$(cat docs/design-notes.md)"
```

## Configuration

`sync` needs no credentials — the registry endpoint is public. `check` and the
`verify_architecture` tool need one API key:

| Variable | Purpose |
| --- | --- |
| `SOTA_ANCHOR_API_KEY` | Preferred. Any OpenAI-compatible key. |
| `SOTA_ANCHOR_BASE_URL` | Defaults to OpenRouter. |
| `SOTA_ANCHOR_MODEL` | Optional. Unset, it is resolved from the live catalog. |
| `OPENROUTER_API_KEY` | Fallback; preferred over `OPENAI_API_KEY`. |
| `OPENAI_API_KEY` | Fallback. |
| `GITHUB_TOKEN` | Optional, raises GitHub search rate limits. |

An unset `SOTA_ANCHOR_MODEL` is resolved from the catalog at runtime: the model
that arbitrates obsolescence should not itself be a stale constant.

## MCP setup

```json
{
  "mcpServers": {
    "sota-anchor": {
      "command": "sota-anchor",
      "args": ["serve"],
      "env": { "SOTA_ANCHOR_API_KEY": "..." }
    }
  }
}
```

Exposes a resource `models://active` (current endpoints and the superseded map,
as JSON), a tool `verify_architecture(pitch)`, and a prompt `init_project`.

---

## Two design decisions worth knowing about

**An empty evidence set can never produce an obsolescence verdict.** If
retrieval returns nothing, the judge is not called at all. A model handed an
empty evidence block will still produce a confident "yes, that's obsolete" from
its own priors, and telling a developer to abandon work they actually need is
the worst thing this tool could do. The empty path returns *"no recent evidence
was retrieved"* — explicitly not *"the limitation still holds"* — and carries any
retrieval errors forward so you can see the difference.

**Queries are AND-joined, with progressive relaxation.** `search_query=all:{phrase}`
is parsed by arXiv as an implicit OR over every word. For one example proposal
that matched **329,590 papers**, and sorted by submission date it returned the
newest arXiv papers about *anything* — robot manipulation, image generation, PDE
surrogates. The arbiter would have judged every proposal against unrelated noise
while appearing to work. AND-joining the same terms returned exactly one paper,
and it was the one that actually bore on the question. Since AND can
over-constrain, both sources walk a ladder from strict to loose and stop at the
first rung that returns anything.

## Known limitations

- **arXiv throttles hard.** It asks for roughly one request every three seconds
  and enforces it by returning `406` with an empty body from its edge. Ladder
  rungs are spaced, capped at three, and retried with backoff, but a busy IP can
  still be turned away — in which case the run reports
  `arxiv: throttled (HTTP 406) ... skipped for this run` and proceeds on GitHub
  evidence alone. Check the reported errors before reading a "not obsolete"
  verdict as reassurance.
- **Verdict quality is bounded by the evidence.** A paper's existence is not
  proof that a production-ready primitive exists. Treat an obsolescence verdict
  as a prompt to go look, not as a decision.
- **The registry occasionally exposes near-duplicate variants** (for example both
  `gemini-3.1-pro-preview` and `gemini-3.1-pro-preview-customtools`), and each
  consumes a slot in the block. Both are genuinely current; it is cosmetic noise.

## On the assertion-reason format

SciUnlearn (Paul, Patwardhan & Cohan, [arXiv:2608.20960](https://arxiv.org/abs/2608.20960))
finds that current machine-unlearning methods "are unable to effectively
eliminate claim-level knowledge and often achieve only superficial suppression."
That is the empirical case for this tool's approach: if outdated claims cannot be
cleanly removed from a model's weights, the correction has to happen **in
context**, at the moment the model is about to act on the stale belief.

Assertion–reason is one of the four QA formats in that benchmark, borrowed here
as the output *shape*. The belief that phrasing an update as assertion plus
reason makes an agent more likely to act on it is this project's design
hypothesis — it is not a result from that paper, and the paper should not be
cited as showing it.

## Development

```bash
uv pip install -e ".[dev]"
python -m pytest            # 231 tests, all offline
```

The suite never touches the network: HTTP is served through
`httpx.MockTransport` and the LLM through an injected fake.

## License

MIT
