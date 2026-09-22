---
description: Refresh the model registry snapshot and the session context block
---

Refresh the model catalog from the public registry and rewrite the local
instruction files. Run the plugin's own checkout, which needs `uv`:

```bash
uv run --quiet --project "${CLAUDE_PLUGIN_ROOT}" sota-anchor sync --target all
```

`--project` keeps the current directory as the working directory, so the files
land in this project. If `uv` is not installed, use `sota-anchor sync --target all`
when `command -v sota-anchor` finds it. If neither resolves, say that the CLI is
unavailable rather than guessing at another command.

This needs no API key. It writes the managed block into `CLAUDE.md`,
`AGENTS.md`, `.cursorrules` and `.cursor/rules/sota.mdc` in this project,
touching nothing outside the `SOTA-ANCHOR` markers, and refreshes the block this
plugin adds at session start.

Then report which endpoints are now current and which superseded identifiers
the block lists, so the user can see what changed. If the command reports
serving a cached catalog, say so: the refresh failed and the data is older than
it looks.

From then on, use the refreshed list as the reference for which model
identifiers the registry currently serves.
