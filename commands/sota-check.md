---
description: Check whether a design relies on a limitation that no longer exists
argument-hint: [design or architecture pitch]
---

Use the `sota-architect` skill to arbitrate this proposal:

$ARGUMENTS

Work through the skill's four steps in order: invert the proposal into an
implicit limitation and a search query, retrieve evidence with
`sota-anchor evidence --query "..." --json`, then judge strictly against what
came back.

If no proposal was supplied above, ask what design to check rather than guessing
from the surrounding conversation.

Report either the `[SOTA ARBITER PARADIGM SHIFT]` block, a plain statement that
the limitation still appears genuine, or - if retrieval returned nothing - that
nothing was checked. Do not treat an empty evidence set as reassurance.
