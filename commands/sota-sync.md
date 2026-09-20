---
description: Refresh the live model catalog and the session context block
---

Refresh the model catalog from the public registry and rewrite the local
instruction files:

```bash
sota-anchor sync --target all
```

This needs no API key. It writes the managed block into `CLAUDE.md`,
`AGENTS.md`, `.cursorrules` and `.cursor/rules/sota.mdc`, touching nothing
outside the `SOTA-ANCHOR` markers, and refreshes the block this plugin injects
at session start.

Then report which endpoints are now active and which superseded identifiers the
block forbids, so the user can see what changed. If the command reports serving
a cached catalog, say so - the refresh failed and the data is older than it
looks.

Afterwards, treat the refreshed list as authoritative over anything you
remember about model names.
