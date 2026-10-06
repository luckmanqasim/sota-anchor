# Privacy

sota-anchor runs entirely on your machine. Its author runs no server, collects nothing,
and receives nothing from it. There is no telemetry, no analytics and no account.

## What leaves your machine

sota-anchor talks only to the public services below, and only to do its job. Each one's
own privacy policy covers what it does with a request.

| Service | When | What it receives |
| --- | --- | --- |
| arXiv, Hugging Face, GitHub, npm, crates.io | When a check runs | The search queries your agent writes from your request, and shorter versions of them |
| GitHub | When a check runs, if you set a GitHub token | That token, in a request header |
| Brave Search | When a check runs, only if you set a Brave key | The search queries and that key |
| OpenRouter (`openrouter.ai/api/v1/models`) | At most once a day, for the model list | A request for its public model list; no key and nothing about you |
| The API at `SOTA_ANCHOR_BASE_URL` | Only if you set `SOTA_ANCHOR_API_KEY` | Your plan, the evidence found for it, and that key, over HTTPS |
| PyPI | On first launch | Requests for the Python packages `uv.lock` pins |

Search queries describe what you're building, so don't use the check on plans you can't
share with those services.

## What stays on your machine

- Optional keys you enter in the plugin's settings, which Claude Code keeps in your
  system's credential store. Each is sent only to its own service, as above.
- The model list and the session block, cached in `~/.cache/sota-anchor`, or in
  `SOTA_ANCHOR_CACHE_DIR`.
- A marked block in `CLAUDE.md`, `AGENTS.md` or Cursor's rules, written only when you run
  `/sota-sync` or `sota-anchor sync`.
- Your prompts. The optional prompt hook, off unless you set `SOTA_ANCHOR_PROMPT_HOOK=1`,
  reads them locally to decide whether to add a nudge, and sends them nowhere.

The full detail is in the README under
[What it sends and fetches](https://github.com/luckmanqasim/sota-anchor#what-it-sends-and-fetches).

## Questions

Open an issue at <https://github.com/luckmanqasim/sota-anchor/issues>.
