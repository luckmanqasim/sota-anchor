sota-anchor plugin: no model registry snapshot yet

No registry snapshot has been fetched, so this session has no current list
of model API endpoints. Model identifiers recalled from training data may
have been superseded since. When code or config needs one, ask the user
which endpoint to use, or run /sota-sync to fetch the current list.

Before building around something assumed to be unavailable - a custom parser
or converter for a format thought to need vendor tooling, a rewrite of an
existing library from scratch, an OCR or heuristic stage around a model
limitation - or before advising that no open-source tool exists for it, run
`/sota-check "<the design>"` or the verify_architecture MCP tool. It
searches recent papers, repositories and package registries for an existing
solution first.
