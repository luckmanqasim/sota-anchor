# sota-anchor

Coding agents carry two kinds of stale knowledge. One is trivia: they write
`gpt-4o` into your `.env` because that was current when they were trained. The
other is expensive: they build a workaround for a limitation that no longer
exists — an OCR-and-snapping pipeline, a hand-rolled retry scheduler, a binary
parser written from scratch for a file format they believe only a vendor SDK
can read — because as far as their weights are concerned, the thing they are
working around is still hard, or still missing.

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
| Superseded endpoint         | Use instead                 |
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

1. **Invert.** Given a proposal, work out what would have to be unavailable,
   impossible or impractical for that design to be justified. Often that is
   not a model limit at all: no reader exists for a format, only a vendor SDK
   can open it, an API does not expose the data. A constraint the user states
   ("I can't use the vendor SDK") is kept as given; what gets checked is the
   assumption underneath it, that nothing else meets that constraint. The
   output is a falsifiable claim plus search queries — reasoned about, not
   looked up in a table.
2. **Differ.** Search work from the past 6–12 months — papers on arXiv and
   Hugging Face, repositories on GitHub, packages on npm and crates.io, and the
   web if a Brave key is set — using *two* vectors: the task in its own
   vocabulary, and whatever would make the workaround unnecessary (an existing
   implementation, library, tool or model capability). A narrow query misses a
   general advance indexed under other terminology; a broad one misses work
   that only names the specific task. This is the deterministic part, and the
   part the plugin does.
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

- **A `SessionStart` hook** that adds a dated snapshot of the model API
  endpoints the registry serves, and the superseded identifiers to prefer them
  over — into *every* session, including one started in an empty directory
  where there is no `CLAUDE.md` to read. Costs ~160 ms, because it reads a
  block rendered ahead of time rather than starting an interpreter or touching
  the network. The block names its source and says what it is for; it makes no
  claim about which model is running and no claim to outrank the model's own
  knowledge. An earlier wording did both, and a host model rightly refused it as
  a prompt injection.
- **A `check-what-exists` skill** that runs the verification protocol.
- **`/sota-check <design>`** and **`/sota-sync`** slash commands.
- **An MCP server** exposing `models://active` and `check_what_exists`.
- **An optional `UserPromptSubmit` hook**, inert unless you set
  `SOTA_ANCHOR_PROMPT_HOOK=1`, which nudges toward verification when a prompt
  asserts that something cannot be done. It only ever adds context — it can
  never block or discard your prompt.

### Zero-key verification

`check_what_exists` is two-phase. Call it with a pitch and it returns an
*inversion prompt*: what would have to be unavailable for this design to be
justified? Answer that, call again with the `domain_query` and
`capability_query` you produced — each led by its most specific term, since
retrieval relaxes a query by dropping its last terms first — and it returns
dated, cited evidence plus the protocol for judging against it.

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
registry and every default evidence source are public — and `check` and
`check_what_exists` fall back to the host-driven protocol instead of failing.
A key is only for a **headless** verdict, where no agent is present to answer
the protocol:

| Variable | Purpose |
| --- | --- |
| `SOTA_ANCHOR_API_KEY` | Preferred. Any OpenAI-compatible key. |
| `SOTA_ANCHOR_BASE_URL` | Defaults to OpenRouter. |
| `SOTA_ANCHOR_MODEL` | Optional. Unset, it is resolved from the live catalog. |
| `OPENROUTER_API_KEY` | Fallback; preferred over `OPENAI_API_KEY`. |
| `OPENAI_API_KEY` | Fallback. |

An unset `SOTA_ANCHOR_MODEL` is resolved from the catalog at runtime: the model
that arbitrates obsolescence should not itself be a stale constant.

Retrieval and the plugin read these, all optional:

| Variable | Purpose |
| --- | --- |
| `BRAVE_API_KEY` | Adds general web search as an evidence source. Unset, the web is simply not queried. |
| `GITHUB_TOKEN` / `GH_TOKEN` | Raises GitHub's unauthenticated limit of ten searches a minute. |
| `SOTA_ANCHOR_CACHE_DIR` | Where the catalog and session block live. Default `~/.cache/sota-anchor`. The hook and the CLI both honour it. |
| `SOTA_ANCHOR_TTL_MINUTES` | How old the session block may get before the hook refreshes it. Default 1440. |
| `SOTA_ANCHOR_PROMPT_HOOK` | `1` switches on the prompt-time nudge. |

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
as JSON), a tool `check_what_exists(pitch, domain_query?, capability_query?,
months?)`, and a prompt `init_project`.

The judging payload states an objective standard rather than a rhetorical one:
retrieved evidence only, with training data inadmissible in either direction;
the assumption stands unless evidence documents supersession; and a threshold
that depends on the kind of claim. "No reader exists for this format" is an
*existence* claim, refuted by a published repository or package that does the
job, reported with its age and activity because existence is not maturity.
"Models cannot do this accurately" is a *performance* claim, and needs
benchmarked results. Frontier grounding comes from a dated registry snapshot of
what current endpoints *declare* they accept and emit — an interface reading,
never a performance claim, and never a hardcoded list of model capabilities that
would itself go stale.

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
over-constrain, every source walks a ladder from strict to loose, dropping the
*last-written* term at each rung and stopping at the first rung that returns
something relevant. Term order is the query author's, not word length: length
dropped short identifiers first, and `nwd reader` finds an independent NWD
reader on GitHub where `navisworks reader parser` finds nothing. Because
semantic and fuzzy search never return empty, a result must also mention two of
its query's terms to count as evidence.

## Known limitations

- **arXiv is fussy about its client.** Its edge answers httpx with `406` where
  it answers curl and urllib with `200`; the discriminator was never isolated
  despite varying headers, encoding, HTTP version and keep-alive. A refused
  request therefore goes straight to urllib, which works — backing off first
  cost 45 seconds of a 55-second retrieval for nothing. Requests are spaced 3.5s
  apart and kept off reused connections, per its Terms of Use. If both paths
  fail the run reports `arxiv: throttled ... skipped for this run` and proceeds
  on the other sources — check the reported errors before reading a "not
  obsolete" verdict as reassurance.
- **PyPI is not searched.** It has no search API, and its search page answers
  clients with a JavaScript challenge. Python packages are usually still found
  through their GitHub repositories.
- **Relevance is lexical.** The two-term floor keeps out a project that merely
  shares an acronym, but not one that shares generic words: a "multimodal RAG
  with vector search" repository passes for "multimodal vector coordinate
  extraction". The judge sees every description and the standard of proof
  rules off-target evidence out, but expect some noise in the evidence list.
- **Retrieval is bounded in time.** All sources share a 60-second deadline and at
  most four requests in flight. A source still running at the deadline is cut
  off, keeps what it found, and says so in the errors.
- **Whether the check runs is the host model's call.** It matches your request
  against the skill's description. Measured headless on the NWD prompts, a
  description that only said "use before building around…" never fired on
  "convert some nwds to glbs, write a custom parser": the model advised against
  the parser from memory instead, and advice was not covered. Naming the advice
  path fixed it (0/3 to 3/3, the check first every time) without firing on
  ordinary requests. The opt-in prompt hook, tested alone, did not help.
- **Verdict quality is bounded by the evidence.** A paper's existence is not
  proof that a production-ready primitive exists, and a repository's is not
  proof that it works. Treat an obsolescence verdict as a prompt to go look, not
  as a decision.
- **The registry occasionally exposes near-duplicate variants** (for example both
  `gemini-3.1-pro-preview` and `gemini-3.1-pro-preview-customtools`), and each
  consumes a slot in the block. Both are genuinely current; it is cosmetic noise.
- **The plugin needs `uv` (or the CLI on `PATH`).** Retrieval is Python; it
  cannot be done from bash. The hook refreshes its block with `uv run --project`
  against the plugin's own checkout, which leaves your project directory
  untouched. Without `uv` or the CLI, the `SessionStart` hook still injects
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
python -m pytest            # all offline
```

The suite never touches the network or your own cache. HTTP is served through
`httpx.MockTransport` and the LLM through an injected fake. Every test gets a
private cache directory, a real DNS lookup fails the test, and the hook tests
put fake refresh tools first on `PATH`. Those guards exist because both leaks
happened: tests once wrote fixture models into the real session block, which
the plugin then injected into live sessions. `ruff check .` is clean and both
run in CI on Linux and Windows.

## License

MIT
