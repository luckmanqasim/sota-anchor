---
description: Check whether a design relies on something assumed to be unavailable that may no longer be
argument-hint: [design or architecture pitch]
---

Use the `sota-architect` skill to check this proposal:

$ARGUMENTS

Work through the skill's steps in order: invert the proposal into the limitation
it assumes plus a `domain_query` and a `capability_query`, retrieve evidence with
the `verify_architecture` MCP tool (the skill's CLI fallback only if the server
is unreachable), then judge strictly against what came back.

The check is not limited to what models can do. A custom parser for a format
thought to need a vendor SDK, or a library rewritten from scratch, rests on an
assumption just as checkable.

If no proposal was supplied above, ask what design to check rather than guessing
from the surrounding conversation.

Report either the `[SOTA ARBITER PARADIGM SHIFT]` block, a plain statement that
the limitation still appears genuine, or - if retrieval returned nothing - that
nothing was checked. Do not treat an empty evidence set as reassurance.
