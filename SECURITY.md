# Security

## Reporting a vulnerability

Report it privately through GitHub: open the repository's
[Security tab](https://github.com/luckmanqasim/sota-anchor/security) and choose
**Report a vulnerability**. Please don't open a public issue for it.

Include what you found, how to reproduce it, and what it lets an attacker do. You'll get
an acknowledgement within a week, and a fix or a decision as soon as the issue is
understood. Fixes ship as a new release on PyPI and in the plugin, and the release notes
credit the reporter unless you'd rather not be named.

## Supported versions

Only the latest release gets security fixes. Update with `uv tool upgrade sota-anchor`,
or by updating the plugin in Claude Code.

## In scope

- The `sota-anchor` package: the MCP server, the CLI and everything under `src/`.
- The Claude Code plugin: its hooks, skill, commands and manifests.
- The release workflow in `.github/workflows/publish.yml`.

Text that sota-anchor retrieves from third-party services (paper abstracts, repository
and package descriptions) is untrusted input. A way for that text to reach the agent as
anything other than quoted data, or to change what sota-anchor itself does, is in scope.

## Out of scope

- Vulnerabilities in the services sota-anchor queries, such as GitHub or arXiv. Report
  those to the service.
- Vulnerabilities in dependencies with no path to exploit them through sota-anchor.
  Report those upstream, though a heads-up here is welcome.
