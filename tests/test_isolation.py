"""The suite must never touch the developer's real cache or the network.

Both happened. The CLI `sync` tests wrote fixture data -- `slowcorp/steady-1`
and nothing else -- into the real ~/.cache/sota-anchor/session-block.md, and
the SessionStart hook injected it into live sessions, where the host model
rightly refused it as a prompt injection. And a hook test's detached refresh
fetched the live OpenRouter catalog. These tests pin the guards in conftest.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest


class TestCacheIsolation:
    def test_every_test_gets_a_private_cache_directory(self):
        from sota_anchor.catalog import default_cache_path

        real = Path.home() / ".cache" / "sota-anchor"
        assert default_cache_path().parent != real

    def test_the_session_block_resolves_inside_it(self):
        from sota_anchor.catalog import default_cache_path
        from sota_anchor.seed import seed_path

        assert seed_path().parent == default_cache_path().parent


class TestNetworkIsolation:
    def test_resolving_a_public_host_fails(self):
        with pytest.raises(RuntimeError, match="network"):
            socket.getaddrinfo("openrouter.ai", 443)

    def test_loopback_still_resolves(self):
        # asyncio and local test servers need loopback; only the outside is fenced.
        assert socket.getaddrinfo("127.0.0.1", 80)
