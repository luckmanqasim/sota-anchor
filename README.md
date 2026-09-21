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

1. **Invert.** Given a proposal, work out what would have to be hard, impossible
   or inaccurate for that design to be justified. The output is a falsifiable
   claim plus a search query — reasoned about, not looked up in a table.
2. **Differ.** Query arXiv and GitHub for work from the past 6–12 months, using
   *two* vectors: the task in its own vocabulary, and the broad capability that
   would make the workaround unnecessary. A narrow query misses a general
   advance indexed under other terminology; a broad one misses work that only
   names the specific task. This is the deterministic part, and the part the
   plugin does.
3. **Judge.** Decide whether the evidence has retired the claim. If it has, the
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

## Install as a Claude Code plugin

This is the primary way to use it, and it needs **no LLM API key**. The host
session does the reasoning; the plugin does the retrieval.

```bash
git clone https://github.com/luckmanqasim/sota-anchor
claude --plugin-dir ./sota-anchor          # try it for one session
claude plugin validate ./sota-anchor       # check the manifests
```

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). The bundled MCP
server runs out of the plugin's own checkout, so there is nothing to `pip
install` first.

What you get:

- **A `SessionStart` hook** that injects the current frontier lineup and the
  superseded identifiers not to reach for — into *every* session, including one
  started in an empty directory where there is no `CLAUDE.md` to read. Costs
  ~160 ms, because it reads a block rendered ahead of time rather than starting
  an interpreter or touching the network.
- **A `sota-architect` skill** that runs the verification protocol.
- **`/sota-check <design>`** and **`/sota-sync`** slash commands.
- **An MCP server** exposing `models://active` and `verify_architecture`.
- **An optional `UserPromptSubmit` hook**, inert unless you set
  `SOTA_ANCHOR_PROMPT_HOOK=1`, which nudges toward verification when a prompt
  asserts that something cannot be done. It only ever adds context — it can
  never block or discard your prompt.

### Zero-key verification

`verify_architecture` is two-phase. Call it with a pitch and it returns an
*inversion prompt*: what would have to be impossible for this design to be
justified? Answer that, call again with the `verification_query` you produced,
and it returns dated, cited evidence plus the protocol for judging against it.

The judging is done by the session you are already in, so there is no second
model and no second bill. Which puts a lot of weight on one rule, stated in the
payload and in the skill: **judge only from the evidence, never from training
data.** The host model is the one carrying the stale priors, so its memory is
not admissible about its own limits.

## Install as a standalone CLI

```bash
uv tool install sota-anchor      # or: uv pip install -e ".[dev]"
```

## Use

```bash
# Refresh instruction files and the session block. Needs no API key.
sota-anchor sync --target all

# Retrieve evidence for a query. Deterministic, no LLM. What the skill calls.
sota-anchor evidence --query "direct vector polygon extraction technical drawings"

# Print the block the SessionStart hook injects; --refresh rewrites the cache.
sota-anchor seed

# Test a proposal.
sota-anchor check "Extract MEP pipe penetrations from PDFs with OCR bounding
                   boxes and geometric snapping heuristics."

# Run as an MCP server over stdio.
sota-anchor serve
```

`sync` options: `--staleness-months` (default 12) for how far behind its own
provider a model may fall before counting as legacy — raise it for providers
that ship slowly; `--provider` (repeatable) and `--all-providers` to widen the
block; `--max-per-provider` (default 4); `--refresh` to bypass the 24-hour cache.

`check` exit codes are meant for CI: **0** the limitation still holds, **2** the
proposal is obsolete, **3** no verdict was reached because no key was configured
and a host model still has to judge. 3 is deliberately not 0 — a gate must not
read "nobody judged this" as "this is fine".

```yaml
- run: sota-anchor check "$(cat docs/design-notes.md)"   # needs a key for 0/2
```

## Configuration

Nothing here needs an API key. `sync`, `seed` and `evidence` never did — the
registry and both evidence sources are public — and `check` and
`verify_architecture` now fall back to the host-driven protocol instead of
failing. A key is only for a **headless** verdict, where no agent is present to
answer the protocol:

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

## MCP setup outside the plugin

The plugin ships its own `.mcp.json`, so this is only for wiring the server into
something else:

```json
{
  "mcpServers": {
    "sota-anchor": {
      "command": "sota-anchor",
      "args": ["serve"]
    }
  }
}
```

Exposes a resource `models://active` (current endpoints and the superseded map,
as JSON), a tool `verify_architecture(pitch, verification_query?, months?)`, and
a prompt `init_project`.

The judging payload states an objective standard rather than a rhetorical one:
retrieved evidence only, with training data inadmissible in either direction;
the assumption stands unless evidence documents supersession; and benchmarked
results preferred over proposed approaches. Frontier grounding comes from a
dated registry snapshot of what current endpoints *declare* they accept and
emit — an interface reading, never a performance claim, and never a hardcoded
list of model capabilities that would itself go stale.

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

- **arXiv is fussy about its client.** Its edge answers httpx with `406` where
  it answers curl and urllib with `200`; the discriminator was never isolated
  despite varying headers, encoding, HTTP version and keep-alive. A refused
  request therefore retries through urllib, which works. Requests are spaced
  3.5s apart and kept off reused connections, per its Terms of Use. If both
  paths fail the run reports `arxiv: throttled ... skipped for this run` and
  proceeds on GitHub alone — check the reported errors before reading a
  "not obsolete" verdict as reassurance.
- **Verdict quality is bounded by the evidence.** A paper's existence is not
  proof that a production-ready primitive exists. Treat an obsolescence verdict
  as a prompt to go look, not as a decision.
- **The registry occasionally exposes near-duplicate variants** (for example both
  `gemini-3.1-pro-preview` and `gemini-3.1-pro-preview-customtools`), and each
  consumes a slot in the block. Both are genuinely current; it is cosmetic noise.
- **The plugin needs `uv` (or the CLI on `PATH`).** Retrieval is Python; it
  cannot be done from bash. Without either, the `SessionStart` hook still injects
  whatever block is cached and then degrades silently, but the MCP server will
  report as failed to connect.
- **The session block can be a day stale.** The hook never blocks on the network:
  it serves the cached block and refreshes out of band, so a lineup that changed
  this morning may not appear until the next session. Run `/sota-sync` to force it.

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
python -m pytest            # 480 tests, all offline
```

The suite never touches the network: HTTP is served through
`httpx.MockTransport` and the LLM through an injected fake.

## License

MIT
