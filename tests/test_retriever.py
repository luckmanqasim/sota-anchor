"""Temporal-gated evidence retrieval.

The regression guarded here is the one that matters most: `all:{query}` with a
multi-word query is parsed by arXiv as an implicit OR across every term. Sorted
by submittedDate that returns the newest papers on arXiv about *anything*
(329,590 matches for the spec's own example), so the arbiter would judge every
proposal against unrelated noise while appearing to work.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from sota_anchor.retriever import (
    ARXIV_URL,
    GITHUB_URL,
    Evidence,
    EvidenceSet,
    build_arxiv_queries,
    build_github_query,
    gather_evidence,
)

from .conftest import NOW

QUERY = "multimodal VLM direct vector polygon extraction technical drawings"

#: The two original sources. Tests of their mechanics name them explicitly, so
#: that sleeps, attempts and errors from the newer sources cannot leak in.
LEGACY = ("arxiv", "github")


@pytest.fixture(autouse=True)
def no_network_fallback(monkeypatch):
    """Keep the arXiv urllib fallback off the network.

    In production a refused httpx request retries through urllib. Left live,
    every test that simulates a 406 would make a real call, so the default is
    replaced with one that refuses. Tests exercising the fallback inject their own.
    """
    import sota_anchor.retriever as retriever

    async def offline(url: str) -> str:
        raise OSError("offline: arXiv fallback disabled in tests")

    monkeypatch.setattr(retriever, "DEFAULT_ARXIV_FALLBACK", offline)


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """Keep throttle delays out of the clock.

    The retriever genuinely waits between arXiv ladder rungs, which is correct in
    production and pure cost in tests. Tests that assert *on* the delays inject
    their own sleeper and are unaffected by this.
    """
    import asyncio

    async def instant(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


def atom(entries: list[tuple[str, str, str]]) -> str:
    """An arXiv Atom feed of (title, published, summary) triples."""
    body = "".join(
        f"""<entry>
    <id>http://arxiv.org/abs/2608.{i:05d}v1</id>
    <title>{title}</title>
    <published>{published}</published>
    <summary>{summary}</summary>
  </entry>"""
        for i, (title, published, summary) in enumerate(entries, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<opensearch:totalResults xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>"
        f"{len(entries)}</opensearch:totalResults>{body}</feed>"
    )


def repos(items: list[tuple[str, str, str]]) -> dict:
    """A GitHub search payload of (full_name, description, pushed_at) triples."""
    return {
        "total_count": len(items),
        "items": [
            {
                "full_name": name,
                "description": description,
                "html_url": f"https://github.com/{name}",
                "pushed_at": pushed_at,
                "stargazers_count": 42,
                "topics": [],
            }
            for name, description, pushed_at in items
        ],
    }


class TestArxivQueryBuilder:
    def test_terms_are_and_joined_not_left_to_implicit_or(self):
        # THE regression. A bare `all:<phrase>` is an OR query in disguise.
        assert " AND " in build_arxiv_queries(QUERY)[0]

    def test_every_term_carries_its_own_field_prefix(self):
        first = build_arxiv_queries("vector polygon extraction")[0]
        assert first.count("all:") == 3

    def test_no_query_in_the_ladder_is_a_bare_unscoped_phrase(self):
        for query in build_arxiv_queries(QUERY):
            terms = query.split(" AND ")
            assert all(term.startswith("all:") for term in terms)
            assert all(" " not in term.removeprefix("all:") for term in terms)

    def test_generic_function_words_are_dropped(self):
        query = build_arxiv_queries("the models cannot extract data from the drawings")[0]
        assert "all:the" not in query
        assert "all:from" not in query
        assert "all:extract" in query

    def test_very_short_tokens_are_dropped(self):
        assert "all:a" not in build_arxiv_queries("a vector polygon extraction")[0]

    def test_three_letter_acronyms_survive_tokenisation(self):
        # Never dropped by the minimum-length filter: short identifiers such as
        # file extensions are often the most specific term in a query.
        from sota_anchor.retriever import extract_terms

        assert "vlm" in extract_terms(QUERY)

    def test_ladder_starts_most_specific(self):
        ladder = build_arxiv_queries(QUERY)
        assert len(ladder[0].split(" AND ")) > len(ladder[-1].split(" AND "))

    def test_ladder_drops_one_term_per_rung(self):
        ladder = build_arxiv_queries("alpha bravo charlie delta")
        assert [len(q.split(" AND ")) for q in ladder] == [4, 3, 2]

    def test_ladder_drops_the_last_written_term_first(self):
        # The widest rung is the first four terms as written; relaxing sheds the
        # fourth ('vector') and keeps the first ('multimodal').
        second = build_arxiv_queries(QUERY)[1]
        assert "all:vector" not in second
        assert "all:multimodal" in second

    def test_ladder_floors_at_two_terms(self):
        assert min(len(q.split(" AND ")) for q in build_arxiv_queries(QUERY)) == 2

    def test_single_term_query_is_still_usable(self):
        assert build_arxiv_queries("bioinformatics") == ["all:bioinformatics"]

    def test_query_of_only_function_words_yields_nothing_to_search(self):
        assert build_arxiv_queries("the and of") == []

    def test_deduplicates_repeated_terms(self):
        assert build_arxiv_queries("vector vector polygon")[0].count("all:vector") == 1

    def test_is_domain_agnostic(self):
        # No topic dictionary: an unrelated discipline must build the same shape.
        query = build_arxiv_queries("ribosome profiling alignment pipeline")[0]
        assert query.count("all:") == 4


class TestGithubQueryBuilder:
    def test_window_is_computed_from_the_current_date(self):
        assert "pushed:>2025-09-19" in build_github_query(QUERY, now=NOW, months=12)

    def test_window_respects_the_month_count(self):
        assert "pushed:>2026-03-19" in build_github_query(QUERY, now=NOW, months=6)

    def test_no_date_is_hardcoded(self):
        # The spec pinned `pushed:>2025-10-01`, ~12 months stale as of today.
        later = build_github_query(QUERY, now=NOW + dt.timedelta(days=400), months=12)
        assert "2025-09-19" not in later

    def test_carries_the_search_terms(self):
        assert "multimodal" in build_github_query(QUERY, now=NOW, months=12)


class TestArxivRetrieval:
    async def _gather(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sources=LEGACY, **kwargs
            )

    async def test_returns_recent_papers_as_evidence(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(
                    200,
                    text=atom([("PlanSightRAG for civil plans", "2026-08-26T10:00:00Z", "Vector extraction from drawings.")]),
                )
            return httpx.Response(200, json=repos([]))

        evidence = await self._gather(handler)
        assert any("PlanSightRAG" in item.title for item in evidence.items)

    async def test_relaxes_the_query_when_the_strictest_returns_nothing(self):
        attempts: list[str] = []

        def handler(request):
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            search = parse_qs(urlparse(str(request.url)).query)["search_query"][0]
            attempts.append(search)
            if len(attempts) < 3:
                return httpx.Response(200, text=atom([]))
            return httpx.Response(
                200, text=atom([("Found late", "2026-08-01T10:00:00Z", "Relevant vector work.")])
            )

        evidence = await self._gather(handler)
        assert len(attempts) == 3
        assert len(attempts[0].split(" AND ")) > len(attempts[2].split(" AND "))
        assert any("Found late" in item.title for item in evidence.items)

    async def test_stops_relaxing_as_soon_as_results_appear(self):
        attempts = 0

        def handler(request):
            nonlocal attempts
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            attempts += 1
            return httpx.Response(
                200, text=atom([("Immediate hit", "2026-09-01T10:00:00Z", "Vector polygons.")])
            )

        await self._gather(handler)
        assert attempts == 1

    async def test_papers_outside_the_window_are_excluded(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(
                    200,
                    text=atom([
                        ("Ancient work", "2019-01-01T10:00:00Z", "Vector extraction drawings."),
                        ("Fresh work", "2026-08-01T10:00:00Z", "Vector extraction drawings."),
                    ]),
                )
            return httpx.Response(200, json=repos([]))

        evidence = await self._gather(handler)
        titles = [item.title for item in evidence.items]
        assert "Fresh work" in titles
        assert "Ancient work" not in titles

    async def test_results_are_reranked_by_term_overlap(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(
                    200,
                    text=atom([
                        ("Barely related", "2026-09-10T10:00:00Z", "A study of vector spaces."),
                        ("Highly related", "2026-08-01T10:00:00Z",
                         "Multimodal VLM direct vector polygon extraction from technical drawings."),
                    ]),
                )
            return httpx.Response(200, json=repos([]))

        evidence = await self._gather(handler)
        assert evidence.items[0].title == "Highly related"

    async def test_result_count_is_capped(self):
        many = [(f"Paper {i}", "2026-08-01T10:00:00Z", "Vector extraction drawings.") for i in range(25)]

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text=atom(many))
            return httpx.Response(200, json=repos([]))

        evidence = await self._gather(handler, per_source=5)
        assert len([i for i in evidence.items if i.source == "arxiv"]) == 5

    async def test_requests_newest_first(self):
        captured: dict[str, list[str]] = {}

        def handler(request):
            if ARXIV_URL in str(request.url):
                captured.update(parse_qs(urlparse(str(request.url)).query))
                return httpx.Response(200, text=atom([("X", "2026-08-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([]))

        await self._gather(handler)
        assert captured["sortBy"] == ["submittedDate"]
        assert captured["sortOrder"] == ["descending"]


class TestGithubRetrieval:
    async def _gather(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sources=LEGACY, **kwargs
            )

    async def test_active_repos_become_evidence(self):
        def handler(request):
            if GITHUB_URL in str(request.url):
                return httpx.Response(
                    200,
                    json=repos([("acme/vlm-plans", "Direct polygon extraction from drawings.", "2026-09-18T09:00:00Z")]),
                )
            return httpx.Response(200, text=atom([]))

        evidence = await self._gather(handler)
        assert any(item.source == "github" for item in evidence.items)
        assert any("acme/vlm-plans" in item.title for item in evidence.items)

    async def test_repos_without_a_description_are_skipped(self):
        def handler(request):
            if GITHUB_URL in str(request.url):
                return httpx.Response(200, json=repos([("acme/empty", None, "2026-09-18T09:00:00Z")]))
            return httpx.Response(200, text=atom([]))

        evidence = await self._gather(handler)
        assert [i for i in evidence.items if i.source == "github"] == []

    async def test_token_is_sent_when_configured(self):
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler, github_token="fake-token-for-tests")
        assert seen.get("authorization") == "Bearer fake-token-for-tests"

    async def test_no_auth_header_without_a_token(self):
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler, github_token=None)
        assert "authorization" not in seen

    async def test_token_comes_from_its_own_variable(self, monkeypatch):
        monkeypatch.setenv("SOTA_ANCHOR_GITHUB_TOKEN", "fake-token-for-tests")
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler)
        assert seen.get("authorization") == "Bearer fake-token-for-tests"

    async def test_a_token_it_was_not_given_is_left_alone(self, monkeypatch):
        # A token already in the environment belongs to the user's other tools.
        # Reading it unasked is what the plugin directory holds a plugin for.
        monkeypatch.setenv("GITHUB_TOKEN", "someone-elses-token")
        monkeypatch.setenv("GH_TOKEN", "someone-elses-token")
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler)
        assert "authorization" not in seen

    async def test_an_unfilled_plugin_setting_counts_as_no_token(self, monkeypatch):
        monkeypatch.setenv("SOTA_ANCHOR_GITHUB_TOKEN", "${user_config.github_token}")
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler)
        assert "authorization" not in seen


class TestDegradation:
    async def _gather(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sources=LEGACY, **kwargs
            )

    async def test_one_source_failing_does_not_lose_the_other(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                raise httpx.ConnectError("arxiv down")
            return httpx.Response(
                200, json=repos([("acme/live", "Polygon extraction.", "2026-09-18T09:00:00Z")])
            )

        evidence = await self._gather(handler)
        assert any(item.source == "github" for item in evidence.items)
        assert evidence.errors

    async def test_http_error_status_is_recorded_not_raised(self):
        def handler(request):
            if GITHUB_URL in str(request.url):
                return httpx.Response(403, json={"message": "rate limited"})
            return httpx.Response(200, text=atom([("X", "2026-08-01T10:00:00Z", "vector")]))

        evidence = await self._gather(handler)
        assert any("github" in error for error in evidence.errors)
        assert any(item.source == "arxiv" for item in evidence.items)

    async def test_both_sources_failing_yields_an_empty_set(self):
        def handler(request):
            raise httpx.ConnectError("offline")

        evidence = await self._gather(handler)
        assert evidence.is_empty is True
        assert len(evidence.errors) == 2

    async def test_malformed_arxiv_xml_is_survived(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text="<not-xml")
            return httpx.Response(200, json=repos([]))

        evidence = await self._gather(handler)
        assert evidence.is_empty is True
        assert evidence.errors


class TestEvidenceRendering:
    def test_empty_set_is_reported_as_empty(self):
        assert EvidenceSet().is_empty is True

    def test_render_states_dates_so_the_judge_can_reason_temporally(self):
        evidence = EvidenceSet(
            items=[
                Evidence(
                    source="arxiv",
                    title="PlanSightRAG",
                    url="http://arxiv.org/abs/2608.00001v1",
                    published=dt.date(2026, 8, 26),
                    snippet="Vector extraction.",
                )
            ]
        )
        rendered = evidence.render()
        assert "2026-08-26" in rendered

    def test_render_attributes_each_item_to_its_source(self):
        evidence = EvidenceSet(
            items=[
                Evidence(
                    source="github",
                    title="acme/plans",
                    url="https://github.com/acme/plans",
                    published=dt.date(2026, 9, 18),
                    snippet="Polygon extraction.",
                )
            ]
        )
        assert "github" in evidence.render()

    def test_render_is_ascii_only(self):
        evidence = EvidenceSet(
            items=[
                Evidence(
                    source="arxiv",
                    title="Café study — results",
                    url="http://arxiv.org/abs/1",
                    published=dt.date(2026, 8, 1),
                    snippet="Naïve baseline.",
                )
            ]
        )
        assert evidence.render().isascii()


class TestTransport:
    def test_default_client_does_not_use_http2(self):
        """HTTP/2 was enabled on a false premise and is now off deliberately.

        It looked like the cure for arXiv's 406 during a session where the real
        variable was an exhausted request quota. The actual cause is connection
        reuse, and HTTP/2 multiplexes onto one connection -- the exact behaviour
        to avoid -- besides making `Connection: close` an illegal header.
        """
        from sota_anchor.retriever import make_client

        client = make_client()
        try:
            assert client._transport._pool._http2 is False
        finally:
            import asyncio

            asyncio.run(client.aclose())

    def test_default_client_follows_redirects(self):
        from sota_anchor.retriever import make_client

        client = make_client()
        try:
            assert client.follow_redirects is True
        finally:
            import asyncio

            asyncio.run(client.aclose())

    def test_default_client_identifies_itself(self):
        from sota_anchor.retriever import make_client

        client = make_client()
        try:
            assert "sota-anchor" in client.headers["User-Agent"]
        finally:
            import asyncio

            asyncio.run(client.aclose())


class TestRateLimiting:
    """arXiv asks for roughly one request every three seconds and enforces it.

    Observed live: after a burst, arXiv answers with 406 (empty body, rejected at
    the Fastly edge) and curl with 429. An unthrottled relaxation ladder fires one
    request per rung back to back, so it trips this on the very first real query
    and the whole arXiv channel goes dark.
    """

    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_consecutive_arxiv_requests_are_spaced(self):
        from sota_anchor.retriever import ARXIV_MIN_INTERVAL

        waits: list[float] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text=atom([]))
            return httpx.Response(200, json=repos([]))

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with self._client(handler) as client:
            await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sleep=fake_sleep, sources=LEGACY
            )

        assert waits
        assert min(waits) >= ARXIV_MIN_INTERVAL

    async def test_a_single_successful_request_is_not_delayed(self):
        waits: list[float] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text=atom([("Hit", "2026-09-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([("a/b", "polygon", "2026-09-01T10:00:00Z")]))

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with self._client(handler) as client:
            await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sleep=fake_sleep, sources=LEGACY
            )

        assert waits == []

    async def test_arxiv_ladder_is_capped(self):
        from sota_anchor.retriever import MAX_ARXIV_ATTEMPTS

        attempts = 0

        def handler(request):
            nonlocal attempts
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            attempts += 1
            return httpx.Response(200, text=atom([]))

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12, sleep=fake_sleep)

        assert attempts == MAX_ARXIV_ATTEMPTS

    async def test_github_ladder_is_capped(self):
        from sota_anchor.retriever import MAX_GITHUB_ATTEMPTS

        attempts = 0

        def handler(request):
            nonlocal attempts
            if GITHUB_URL not in str(request.url):
                return httpx.Response(200, text=atom([]))
            attempts += 1
            return httpx.Response(200, json=repos([]))

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12, sleep=fake_sleep)

        assert attempts == MAX_GITHUB_ATTEMPTS

    async def test_throttled_response_is_retried_then_succeeds(self):
        calls = 0

        def handler(request):
            nonlocal calls
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            calls += 1
            if calls == 1:
                # 429 means "slow down" and does yield to waiting. (A 406 is
                # arXiv refusing the client, and goes to the fallback instead.)
                return httpx.Response(429, text="")
            return httpx.Response(
                200, text=atom([("After backoff", "2026-09-01T10:00:00Z", "vector polygon")])
            )

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            evidence = await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sleep=fake_sleep
            )

        assert any(item.title == "After backoff" for item in evidence.items)

    async def test_retry_waits_longer_each_time(self):
        waits: list[float] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(429, text="")
            return httpx.Response(200, json=repos([]))

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with self._client(handler) as client:
            await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sleep=fake_sleep, sources=("arxiv",)
            )

        assert waits == sorted(waits)
        assert len(set(waits)) > 1

    async def test_persistent_throttling_is_reported_not_raised(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(429, text="")
            return httpx.Response(
                200, json=repos([("acme/live", "polygon extraction", "2026-09-18T09:00:00Z")])
            )

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            evidence = await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sleep=fake_sleep
            )

        assert any("arxiv" in error for error in evidence.errors)
        assert any(item.source == "github" for item in evidence.items)


class TestGithubRelaxation:
    """GitHub ANDs its terms too, and repo descriptions are far shorter than abstracts.

    Live counts for the spec's MEP example: 6 terms -> 0 results, 4 -> 0, 3 -> 0,
    2 -> 41. Without a ladder the GitHub channel is silently always empty.
    """

    def test_ladder_starts_narrow_and_widens(self):
        from sota_anchor.retriever import build_github_queries

        ladder = build_github_queries(QUERY, now=NOW, months=12)
        counts = [len(q.split()) for q in ladder]
        assert counts == sorted(counts, reverse=True)
        assert len(ladder) > 1

    def test_every_rung_keeps_the_temporal_bound(self):
        from sota_anchor.retriever import build_github_queries

        for query in build_github_queries(QUERY, now=NOW, months=12):
            assert "pushed:>2025-09-19" in query

    def test_ladder_reaches_a_single_term(self):
        from sota_anchor.retriever import build_github_queries

        ladder = build_github_queries(QUERY, now=NOW, months=12)
        terms = [q for q in ladder if len(q.split()) == 2]  # one term + pushed bound
        assert terms

    def test_first_written_term_survives_every_rung(self):
        from sota_anchor.retriever import build_github_queries

        for query in build_github_queries(QUERY, now=NOW, months=12):
            assert "multimodal" in query

    async def test_relaxes_when_the_narrow_query_finds_nothing(self):
        attempts: list[str] = []

        def handler(request):
            if GITHUB_URL not in str(request.url):
                return httpx.Response(200, text=atom([]))
            attempts.append(parse_qs(urlparse(str(request.url)).query)["q"][0])
            if len(attempts) < 2:
                return httpx.Response(200, json=repos([]))
            return httpx.Response(
                200,
                json=repos([("acme/found", "Polygon extraction.", "2026-09-18T09:00:00Z")]),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            evidence = await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sources=LEGACY
            )

        assert len(attempts) == 2
        assert len(attempts[0].split()) > len(attempts[1].split())
        assert any(item.title == "acme/found" for item in evidence.items)

    async def test_stops_at_the_first_rung_with_results(self):
        attempts = 0

        def handler(request):
            nonlocal attempts
            if GITHUB_URL not in str(request.url):
                return httpx.Response(200, text=atom([]))
            attempts += 1
            return httpx.Response(
                200, json=repos([("acme/hit", "Polygon extraction.", "2026-09-18T09:00:00Z")])
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12)

        assert attempts == 1


class TestThrottleReporting:
    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_throttling_is_reported_in_plain_language(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)

        error = next(e for e in evidence.errors if e.startswith("arxiv"))
        assert "throttled" in error.lower()
        assert "HTTPStatusError" not in error

    async def test_throttle_message_says_what_was_lost(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(429, text="")
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)

        error = next(e for e in evidence.errors if e.startswith("arxiv"))
        assert "skipped" in error.lower()

    async def test_a_genuine_error_is_not_mislabelled_as_throttling(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(404, text="")
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)

        error = next(e for e in evidence.errors if e.startswith("arxiv"))
        assert "throttled" not in error.lower()

    async def test_the_other_source_still_contributes_while_throttled(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(
                200, json=repos([("acme/live", "polygon extraction", "2026-09-18T09:00:00Z")])
            )

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)

        assert evidence.is_empty is False


class TestArxivConnectionHandling:
    """arXiv answers 406 to a request on a reused keep-alive connection.

    Measured against the live endpoint with everything else held identical:
    a shared keep-alive client gets 406 with sort parameters present, while a
    fresh connection per request gets 200 every time. It matches arXiv's Terms
    of Use, which ask clients to "limit requests to a single connection at a
    time" -- and which specify no User-Agent requirement at all, so the
    User-Agent is good citizenship rather than the fix.
    """

    def _client(self):
        from sota_anchor.retriever import make_client

        return make_client()

    def test_client_disables_keepalive(self):
        import asyncio

        client = self._client()
        try:
            assert client._transport._pool._max_keepalive_connections == 0
        finally:
            asyncio.run(client.aclose())

    def test_user_agent_identifies_the_client(self):
        """Names the tool and where to find it, with no personal address -- the
        repository is public, and arXiv's Terms of Use require no User-Agent.
        """
        import asyncio

        client = self._client()
        try:
            agent = client.headers["User-Agent"]
            assert "sota-anchor" in agent
            assert "github.com" in agent
            assert "mailto:" not in agent
        finally:
            asyncio.run(client.aclose())

    async def test_arxiv_requests_ask_the_connection_to_close(self):
        seen: list[str] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                seen.append(request.headers.get("connection", ""))
                return httpx.Response(200, text=atom([("X", "2026-08-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([]))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12)

        assert seen and all(value.lower() == "close" for value in seen)

    def test_spacing_respects_the_published_rate_limit(self):
        from sota_anchor.retriever import ARXIV_MIN_INTERVAL

        # arXiv asks for no more than one request every three seconds.
        assert ARXIV_MIN_INTERVAL >= 3.5


class TestMultiVectorRetrieval:
    """A single hyper-specific query misses broad capability leaps indexed under
    different terminology, so retrieval takes a ladder of query vectors.
    """

    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_accepts_several_query_vectors(self):
        asked: list[str] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                search = parse_qs(urlparse(str(request.url)).query)["search_query"][0]
                asked.append(search)
                return httpx.Response(200, text=atom([("Hit", "2026-08-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            await gather_evidence(
                ["floorplan room polygon segmentation", "multimodal polygon grounding"],
                client=client, now=NOW, months=12,
            )

        assert any("floorplan" in q for q in asked)
        assert any("grounding" in q for q in asked)

    async def test_a_single_string_still_works(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text=atom([("Hit", "2026-08-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)
        assert evidence.items

    async def test_results_are_deduplicated_across_vectors(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(
                    200, text=atom([("Same paper", "2026-08-01T10:00:00Z", "vector polygon")])
                )
            return httpx.Response(200, json=repos([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence(
                ["query one alpha", "query two beta"], client=client, now=NOW, months=12
            )

        urls = [item.url for item in evidence.items]
        assert len(urls) == len(set(urls))

    async def test_arxiv_budget_does_not_grow_without_bound(self):
        from sota_anchor.retriever import ARXIV_TOTAL_CAP

        attempts = 0

        def handler(request):
            nonlocal attempts
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            attempts += 1
            return httpx.Response(200, text=atom([]))

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(
                ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"],
                client=client, now=NOW, months=12, sleep=fake_sleep,
            )

        # The budget scales with vector count but stops at a ceiling: three
        # vectors must not mean three full ladders against a throttled host.
        assert attempts <= ARXIV_TOTAL_CAP

    async def test_arxiv_vectors_are_queried_sequentially(self):
        # "limit requests to a single connection at a time" -- never concurrent.
        concurrent = 0
        peak = 0

        async def handler(request):
            nonlocal concurrent, peak
            if ARXIV_URL in str(request.url):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0)
                concurrent -= 1
                return httpx.Response(200, text=atom([]))
            return httpx.Response(200, json=repos([]))

        import asyncio

        async def fake_sleep(seconds: float) -> None:
            return None

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await gather_evidence(
                ["alpha beta", "gamma delta"], client=client, now=NOW, months=12, sleep=fake_sleep
            )
        assert peak <= 1

    async def test_an_empty_vector_list_retrieves_nothing(self):
        called = False

        def handler(request):
            nonlocal called
            called = True
            return httpx.Response(200, text=atom([]))

        async with self._client(handler) as client:
            evidence = await gather_evidence([], client=client, now=NOW, months=12)

        assert evidence.is_empty
        assert called is False


class TestArxivFallback:
    """arXiv's edge rejects httpx with 406 where curl and urllib get 200.

    Measured repeatedly on a clean request quota, with headers, percent-encoding,
    HTTP version, redirect handling and keep-alive all varied: httpx is refused
    and urllib is not. The discriminator was never isolated. Since arXiv is the
    source that carries benchmark depth, and GitHub descriptions alone cannot
    show a capability leap, a refused httpx request falls back to urllib rather
    than letting the academic channel go dark.
    """

    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_a_refused_request_falls_back(self):
        used: list[str] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(200, json=repos([]))

        async def fallback(url: str) -> str:
            used.append(url)
            return atom([("Rescued paper", "2026-08-26T10:00:00Z", "vector polygon extraction")])

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            evidence = await gather_evidence(
                QUERY, client=client, now=NOW, months=12,
                sleep=fake_sleep, arxiv_fallback=fallback,
            )

        assert used, "fallback was never attempted"
        assert any(item.title == "Rescued paper" for item in evidence.items)

    async def test_a_refusal_goes_straight_to_the_fallback(self):
        # httpx is refused with 406 however often it retries. Measured live, each
        # rung spent 5s + 10s of backoff before reaching the fallback: 45s of a
        # 55s retrieval, against a 60s deadline shared with every other source.
        calls = 0
        waits: list[float] = []

        def handler(request):
            nonlocal calls
            if ARXIV_URL in str(request.url):
                calls += 1
                return httpx.Response(406, text="")
            return httpx.Response(200, json=repos([]))

        async def fallback(url: str) -> str:
            return atom([("Rescued", "2026-08-26T10:00:00Z", "vector polygon extraction")])

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        async with self._client(handler) as client:
            evidence = await gather_evidence(
                QUERY, client=client, now=NOW, months=12, sources=("arxiv",),
                sleep=fake_sleep, arxiv_fallback=fallback,
            )

        assert calls == 1
        assert waits == []
        assert [item.title for item in evidence.items] == ["Rescued"]

    async def test_the_fallback_url_percent_encodes_spaces(self):
        # arXiv reads '+' as a literal plus, which makes the field query invalid.
        captured: list[str] = []

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(200, json=repos([]))

        async def fallback(url: str) -> str:
            captured.append(url)
            return atom([])

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12,
                                  sleep=fake_sleep, arxiv_fallback=fallback)

        assert captured
        assert "%20" in captured[0]
        assert "+" not in captured[0].split("search_query=")[1].split("&")[0]

    async def test_a_healthy_request_never_reaches_the_fallback(self):
        used = False

        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(200, text=atom([("Direct", "2026-08-01T10:00:00Z", "vector")]))
            return httpx.Response(200, json=repos([]))

        async def fallback(url: str) -> str:
            nonlocal used
            used = True
            return atom([])

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12,
                                             arxiv_fallback=fallback)

        assert used is False
        assert any(item.title == "Direct" for item in evidence.items)

    async def test_a_failing_fallback_still_reports_throttling(self):
        def handler(request):
            if ARXIV_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(200, json=repos([]))

        async def fallback(url: str) -> str:
            raise OSError("urllib also refused")

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12,
                                             sleep=fake_sleep, arxiv_fallback=fallback)

        assert any("arxiv" in error for error in evidence.errors)

    async def test_github_has_no_such_fallback(self):
        # Only arXiv exhibits the refusal; GitHub must not grow a second path.
        used = False

        def handler(request):
            if GITHUB_URL in str(request.url):
                return httpx.Response(406, text="")
            return httpx.Response(200, text=atom([]))

        async def fallback(url: str) -> str:
            nonlocal used
            used = True
            return atom([])

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(QUERY, client=client, now=NOW, months=12,
                                  sleep=fake_sleep, arxiv_fallback=fallback)
        assert used is False


class TestLadderWidth:
    """A six-term AND is effectively guaranteed empty on arXiv.

    Observed: a 4-term AND returned the one on-target paper for a real proposal,
    while 6 terms returned nothing. Starting the ladder at full width spends the
    most expensive attempt on a query that cannot match, and with two vectors
    sharing a budget the ladder never reaches a width that would.
    """

    def test_ladder_starts_no_wider_than_the_cap(self):
        from sota_anchor.retriever import ARXIV_MAX_TERMS

        ladder = build_arxiv_queries(
            "MEP pipe penetration extraction construction drawings multimodal"
        )
        assert len(ladder[0].split(" AND ")) <= ARXIV_MAX_TERMS

    def test_the_cap_keeps_the_first_written_terms(self):
        first = build_arxiv_queries(
            "MEP pipe penetration extraction construction drawings multimodal"
        )[0]
        # Length used to decide, and it shed "mep" -- the most specific term in
        # the query -- before generic words like "construction".
        assert "all:mep" in first
        assert "all:construction" not in first

    def test_a_short_query_is_unaffected(self):
        ladder = build_arxiv_queries("vector polygon extraction")
        assert len(ladder[0].split(" AND ")) == 3

    def test_ladder_still_relaxes_to_the_floor(self):
        ladder = build_arxiv_queries(
            "MEP pipe penetration extraction construction drawings multimodal"
        )
        assert min(len(q.split(" AND ")) for q in ladder) == 2


class TestBudgetScaling:
    """The budget scales with vector count, up to a ceiling.

    A flat shared budget was too shallow: three attempts split across two
    vectors let neither relax past its first rung, so a first-rung miss meant no
    evidence at all.
    """

    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def _attempts(self, vectors) -> int:
        count = 0

        def handler(request):
            nonlocal count
            if ARXIV_URL not in str(request.url):
                return httpx.Response(200, json=repos([]))
            count += 1
            return httpx.Response(200, text=atom([]))

        async def fake_sleep(seconds: float) -> None:
            return None

        async with self._client(handler) as client:
            await gather_evidence(vectors, client=client, now=NOW, months=12, sleep=fake_sleep)
        return count

    async def test_two_vectors_get_more_attempts_than_one(self):
        one = await self._attempts(["alpha bravo charlie delta"])
        two = await self._attempts(["alpha bravo charlie delta", "echo foxtrot golf hotel"])
        assert two > one

    async def test_each_vector_can_still_relax(self):
        # Two vectors, each needing at least a second rung to be useful.
        attempts = await self._attempts(
            ["alpha bravo charlie delta", "echo foxtrot golf hotel"]
        )
        assert attempts >= 4

    async def test_total_is_capped_however_many_vectors(self):
        from sota_anchor.retriever import ARXIV_TOTAL_CAP

        attempts = await self._attempts(
            ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima"]
        )
        assert attempts <= ARXIV_TOTAL_CAP


# ---------------------------------------------------------------------------
# Multi-source retrieval: package registries, Hugging Face papers, the optional
# web source, and the budgets every source shares.
# ---------------------------------------------------------------------------

HOSTS = {
    "export.arxiv.org": "arxiv",
    "api.github.com": "github",
    "huggingface.co": "huggingface",
    "registry.npmjs.org": "npm",
    "crates.io": "crates",
    "api.search.brave.com": "web",
}

KEYLESS = ("arxiv", "github", "huggingface", "npm", "crates")


def empty(source: str) -> httpx.Response:
    """A well-formed response with no results, in each source's own shape."""
    if source == "arxiv":
        return httpx.Response(200, text=atom([]))
    bodies = {
        "github": repos([]),
        "huggingface": [],
        "npm": {"objects": [], "total": 0},
        "crates": {"crates": [], "meta": {"total": 0}},
        "web": {"type": "search", "web": {"type": "search", "results": []}},
    }
    return httpx.Response(200, json=bodies[source])


def router(**handlers):
    """Dispatch by host to per-source handlers; an unrouted source answers empty.

    An unknown host raises, so a request to anywhere unexpected fails the test.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        source = HOSTS[request.url.host]
        chosen = handlers.get(source)
        if chosen is None:
            return empty(source)
        result = chosen(request)
        if inspect.isawaitable(result):
            result = await result
        return result

    return handler


def query_of(request: httpx.Request, key: str) -> str:
    return parse_qs(urlparse(str(request.url)).query)[key][0]


def hf_papers(items: list[tuple[str, str, str, str]], **extra) -> list[dict]:
    """A Hugging Face paper-search payload of (arxiv_id, title, publishedAt, summary)."""
    return [
        {
            "paper": {
                "id": arxiv_id,
                "title": title,
                "summary": summary,
                "publishedAt": published,
                "upvotes": 12,
                **extra,
            },
            "publishedAt": published,
            "title": title,
            "summary": summary,
        }
        for arxiv_id, title, published, summary in items
    ]


def npm_packages(items: list[tuple[str, str, str]]) -> dict:
    """An npm registry search payload of (name, description, date)."""
    return {
        "objects": [
            {
                "package": {
                    "name": name,
                    "version": "1.4.0",
                    "description": description,
                    "keywords": [],
                    "date": date,
                    "links": {"npm": f"https://www.npmjs.com/package/{name}"},
                }
            }
            for name, description, date in items
        ],
        "total": len(items),
    }


def crate_list(items: list[tuple[str, str, str]]) -> dict:
    """A crates.io search payload of (name, description, updated_at)."""
    return {
        "crates": [
            {
                "name": name,
                "description": description,
                "updated_at": updated,
                "max_stable_version": "0.3.1",
                "downloads": 12345,
            }
            for name, description, updated in items
        ],
        "meta": {"total": len(items)},
    }


def web_results(items: list[tuple[str, str, str, str | None]]) -> dict:
    """A Brave web-search payload of (title, url, description, page_age)."""
    return {
        "type": "search",
        "web": {
            "type": "search",
            "results": [
                {"title": title, "url": url, "description": description, "page_age": page_age}
                for title, url, description, page_age in items
            ],
        },
    }


async def collect(text, handler, **kwargs) -> EvidenceSet:
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("months", 12)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await gather_evidence(text, client=client, **kwargs)


class TestAuthorOrderSalience:
    """The query's author decides which terms are specific, not their length.

    Length salience dropped the shortest terms first, and in an engineering
    query the shortest terms are usually the identifiers: `nwd`, `glb`, `mep`.
    On live GitHub, `nwd reader` found the independent NWD reader a refusing
    session never looked for, while `navisworks reader parser` found nothing.
    """

    def test_terms_keep_the_order_they_were_written(self):
        from sota_anchor.retriever import extract_terms

        assert extract_terms("nwd navisworks reader parser") == [
            "nwd",
            "navisworks",
            "reader",
            "parser",
        ]

    def test_function_words_are_still_dropped(self):
        from sota_anchor.retriever import extract_terms

        assert extract_terms("a reader for the nwd format") == ["reader", "nwd", "format"]

    def test_a_leading_identifier_survives_every_arxiv_rung(self):
        for query in build_arxiv_queries("nwd navisworks reader parser"):
            assert "all:nwd" in query

    def test_a_leading_identifier_survives_every_github_rung(self):
        from sota_anchor.retriever import build_github_queries

        for query in build_github_queries("nwd navisworks reader parser", now=NOW, months=12):
            assert query.split()[0] == "nwd"


class TestGithubBudget:
    """Three attempts shared by two vectors ran out before either reached its
    two-term rung, which is where repository descriptions start to match."""

    async def _asked(self, vectors) -> list[str]:
        asked: list[str] = []

        def github(request):
            asked.append(query_of(request, "q"))
            return httpx.Response(200, json=repos([]))

        await collect(vectors, router(github=github), sources=("github",))
        return asked

    async def test_each_vector_reaches_its_two_term_rung(self):
        asked = await self._asked(["nwd glb conversion", "nwd navisworks reader"])
        two_terms = [q for q in asked if len(q.split()) == 3]  # plus the pushed bound
        assert any(q.startswith("nwd glb ") for q in two_terms)
        assert any(q.startswith("nwd navisworks ") for q in two_terms)

    async def test_the_widest_rung_is_three_terms(self):
        asked = await self._asked(["nwd navisworks reader parser"])
        assert len(asked[0].split()) == 4  # three terms plus the pushed bound

    async def test_total_is_capped_however_many_vectors(self):
        from sota_anchor.retriever import GITHUB_TOTAL_CAP

        asked = await self._asked(
            ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima"]
        )
        assert len(asked) <= GITHUB_TOTAL_CAP


class TestRelevanceFloor:
    """Fuzzy and semantic search never come back empty, and a one-word GitHub
    query matches every project that shares an acronym. A result must mention
    two of its vector's terms to count as evidence."""

    async def test_an_item_matching_one_term_is_not_evidence(self):
        def github(request):
            if query_of(request, "q").split()[:-1] == ["nwd"]:
                return httpx.Response(
                    200,
                    json=repos(
                        [
                            (
                                "jwwangchn/NWD",
                                "Normalized Gaussian Wasserstein Distance for tiny objects",
                                "2026-09-01T00:00:00Z",
                            )
                        ]
                    ),
                )
            return httpx.Response(200, json=repos([]))

        evidence = await collect(["nwd converter"], router(github=github), sources=("github",))
        assert evidence.is_empty

    async def test_an_off_topic_rung_does_not_stop_the_ladder(self):
        asked: list[str] = []

        def github(request):
            asked.append(query_of(request, "q"))
            if len(asked) == 1:
                noise = ("acme/noise", "Unrelated nwd utilities", "2026-09-01T00:00:00Z")
                return httpx.Response(200, json=repos([noise]))
            hit = ("acme/nwd-reader", "Reads NWD files", "2026-09-01T00:00:00Z")
            return httpx.Response(200, json=repos([hit]))

        evidence = await collect(["nwd reader"], router(github=github), sources=("github",))
        assert len(asked) == 2
        assert [item.title for item in evidence.items] == ["acme/nwd-reader"]

    async def test_a_single_term_vector_needs_only_that_term(self):
        def github(request):
            hit = ("acme/ifc-tools", "Helpers around IFC models", "2026-09-01T00:00:00Z")
            return httpx.Response(200, json=repos([hit]))

        evidence = await collect(["ifc"], router(github=github), sources=("github",))
        assert evidence.items

    async def test_arxiv_keeps_its_keyword_and_semantics(self):
        # arXiv ANDs every term already, and matches stemmed forms the overlap
        # count cannot see; a floor there would reject on-target papers.
        def arxiv(request):
            return httpx.Response(200, text=atom([("X", "2026-08-01T10:00:00Z", "vector")]))

        evidence = await collect(QUERY, router(arxiv=arxiv), sources=("arxiv",))
        assert [item.title for item in evidence.items] == ["X"]


class TestHuggingFacePapers:
    """Semantic search over papers: what arXiv's keyword AND cannot do."""

    VECTOR = "vector coordinate extraction engineering drawings"

    @staticmethod
    def _paper(**extra):
        return hf_papers(
            [
                (
                    "2601.12345",
                    "Vector extraction from engineering drawings",
                    "2026-01-23T00:00:00.000Z",
                    "We extract vector coordinates from drawings directly.",
                )
            ],
            **extra,
        )

    async def test_papers_become_dated_evidence(self):
        def hf(request):
            return httpx.Response(200, json=self._paper())

        evidence = await collect(self.VECTOR, router(huggingface=hf), sources=("huggingface",))
        [item] = evidence.items
        assert item.source == "huggingface"
        assert item.url == "https://huggingface.co/papers/2601.12345"
        assert item.published == dt.date(2026, 1, 23)

    async def test_searches_with_the_whole_phrase(self):
        seen: list[str] = []

        def hf(request):
            seen.append(query_of(request, "q"))
            return httpx.Response(200, json=[])

        await collect(self.VECTOR, router(huggingface=hf), sources=("huggingface",))
        assert seen == [self.VECTOR]

    async def test_one_request_per_vector(self):
        seen: list[str] = []

        def hf(request):
            seen.append(query_of(request, "q"))
            return httpx.Response(200, json=[])

        await collect(
            ["alpha bravo", "charlie delta"], router(huggingface=hf), sources=("huggingface",)
        )
        assert seen == ["alpha bravo", "charlie delta"]

    async def test_papers_outside_the_window_are_excluded(self):
        def hf(request):
            return httpx.Response(
                200,
                json=hf_papers(
                    [
                        (
                            "2410.00001",
                            "Vector extraction from engineering drawings",
                            "2024-10-02T00:00:00.000Z",
                            "Vector coordinates from drawings.",
                        )
                    ]
                ),
            )

        evidence = await collect(self.VECTOR, router(huggingface=hf), sources=("huggingface",))
        assert evidence.is_empty

    async def test_off_topic_semantic_matches_are_dropped(self):
        # Observed live: "navisworks nwd reader" returned marine-fog nowcasting.
        def hf(request):
            fog = ("2603.00001", "Generative nowcasting of marine fog", "2026-03-24T00:00:00Z", "Visibility.")
            return httpx.Response(200, json=hf_papers([fog]))

        evidence = await collect(
            "nwd navisworks reader", router(huggingface=hf), sources=("huggingface",)
        )
        assert evidence.is_empty

    async def test_reports_upvotes_and_linked_code(self):
        def hf(request):
            return httpx.Response(200, json=self._paper(githubRepo="https://github.com/acme/vecx"))

        evidence = await collect(self.VECTOR, router(huggingface=hf), sources=("huggingface",))
        assert "12 upvotes" in evidence.items[0].detail
        assert "https://github.com/acme/vecx" in evidence.items[0].detail

    async def test_a_paper_arxiv_also_returned_appears_once(self):
        title = "Vector extraction from engineering drawings"
        summary = "Vector coordinate extraction from engineering drawings."

        def arxiv(request):
            return httpx.Response(200, text=atom([(title, "2026-08-26T10:00:00Z", summary)]))

        def hf(request):
            paper = ("2608.00001", title, "2026-08-26T00:00:00.000Z", summary)
            return httpx.Response(200, json=hf_papers([paper]))

        evidence = await collect(
            self.VECTOR, router(arxiv=arxiv, huggingface=hf), sources=("arxiv", "huggingface")
        )
        assert len(evidence.items) == 1

    async def test_an_unexpected_payload_is_reported_not_raised(self):
        def hf(request):
            return httpx.Response(200, json={"error": "the shape changed"})

        evidence = await collect(self.VECTOR, router(huggingface=hf), sources=("huggingface",))
        assert any(error.startswith("huggingface") for error in evidence.errors)


class TestNpmRegistry:
    async def test_packages_become_evidence(self):
        def npm(request):
            package = (
                "gltf-pipeline",
                "Content pipeline tools for glTF assets, with a converter.",
                "2026-04-03T00:00:00.000Z",
            )
            return httpx.Response(200, json=npm_packages([package]))

        evidence = await collect("gltf converter", router(npm=npm), sources=("npm",))
        [item] = evidence.items
        assert item.source == "npm"
        assert item.url == "https://www.npmjs.com/package/gltf-pipeline"
        assert item.detail == "version 1.4.0"

    async def test_relaxes_the_search_text_rung_by_rung(self):
        seen: list[str] = []

        def npm(request):
            seen.append(query_of(request, "text"))
            return httpx.Response(200, json=npm_packages([]))

        await collect("nwd glb converter", router(npm=npm), sources=("npm",))
        assert seen[:2] == ["nwd glb converter", "nwd glb"]

    async def test_releases_outside_the_window_are_excluded(self):
        def npm(request):
            old = ("gltf-legacy", "An old glTF converter.", "2019-05-05T00:00:00.000Z")
            return httpx.Response(200, json=npm_packages([old]))

        evidence = await collect("gltf converter", router(npm=npm), sources=("npm",))
        assert evidence.is_empty

    async def test_fuzzy_matches_below_the_floor_are_dropped(self):
        def npm(request):
            namesake = ("nwd", "Selenium WebDriver wire protocol for node.", "2026-06-09T00:00:00Z")
            return httpx.Response(200, json=npm_packages([namesake]))

        evidence = await collect("nwd navisworks reader", router(npm=npm), sources=("npm",))
        assert evidence.is_empty


class TestCratesRegistry:
    async def test_crates_become_evidence(self):
        def crates(request):
            crate = ("gltf-convert", "A glTF converter for Rust.", "2026-09-13T10:00:00.000000Z")
            return httpx.Response(200, json=crate_list([crate]))

        evidence = await collect("gltf converter", router(crates=crates), sources=("crates",))
        [item] = evidence.items
        assert item.source == "crates"
        assert item.url == "https://crates.io/crates/gltf-convert"
        assert "version 0.3.1" in item.detail
        assert "12,345 downloads" in item.detail

    async def test_requests_identify_the_client(self):
        # crates.io's crawler policy requires a User-Agent naming the client.
        agents: list[str] = []

        def crates(request):
            agents.append(request.headers.get("user-agent", ""))
            return httpx.Response(200, json=crate_list([]))

        await collect("gltf converter", router(crates=crates), sources=("crates",))
        assert agents and all("sota-anchor" in agent for agent in agents)

    async def test_requests_are_spaced_to_the_crawler_policy(self):
        waits: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            waits.append(seconds)

        await collect(
            ["alpha bravo charlie", "delta echo foxtrot"],
            router(),
            sources=("crates",),
            sleep=fake_sleep,
        )
        assert waits
        assert min(waits) >= 1.0


class TestWebSearch:
    """Optional: a general web source, active only when SOTA_ANCHOR_BRAVE_API_KEY
    is set, so the zero-key default never changes."""

    async def test_is_skipped_without_a_key(self):
        called: list[str] = []

        def web(request):
            called.append(str(request.url))
            return httpx.Response(200, json=web_results([]))

        evidence = await collect("nwd reader", router(web=web))
        assert called == []
        assert not any(error.startswith("web") for error in evidence.errors)

    async def test_runs_when_its_own_variable_holds_a_key(self, monkeypatch):
        monkeypatch.setenv("SOTA_ANCHOR_BRAVE_API_KEY", "test-key-not-real")
        tokens: list[str] = []

        def web(request):
            tokens.append(request.headers.get("x-subscription-token", ""))
            return httpx.Response(200, json=web_results([]))

        await collect("nwd reader", router(web=web))
        assert tokens and set(tokens) == {"test-key-not-real"}

    async def test_a_key_it_was_not_given_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "someone-elses-key")
        called: list[str] = []

        def web(request):
            called.append(str(request.url))
            return httpx.Response(200, json=web_results([]))

        await collect("nwd reader", router(web=web))
        assert called == []

    async def test_freshness_is_the_rolling_window(self):
        windows: list[str] = []

        def web(request):
            windows.append(query_of(request, "freshness"))
            return httpx.Response(200, json=web_results([]))

        await collect("nwd reader", router(web=web), sources=("web",), web_api_key="k")
        assert windows == ["2025-09-19to2026-09-19"]

    async def test_results_are_dated_by_page_age_and_undated_ones_skipped(self):
        def web(request):
            return httpx.Response(
                200,
                json=web_results(
                    [
                        (
                            "An open NWD reader",
                            "https://example.com/nwd-reader",
                            "Read <strong>NWD</strong> files with this reader.",
                            "2026-09-10T08:00:00",
                        ),
                        ("Undated", "https://example.com/other", "Another nwd reader.", None),
                    ]
                ),
            )

        evidence = await collect("nwd reader", router(web=web), sources=("web",), web_api_key="k")
        [item] = evidence.items
        assert item.published == dt.date(2026, 9, 10)
        assert "<strong>" not in item.snippet

    async def test_requested_without_a_key_it_says_why(self):
        evidence = await collect("nwd reader", router(), sources=("web",))
        [error] = evidence.errors
        assert "SOTA_ANCHOR_BRAVE_API_KEY" in error

    async def test_a_rejected_key_is_reported_without_echoing_it(self):
        def web(request):
            return httpx.Response(401, json={"error": "unauthorized"})

        evidence = await collect(
            "nwd reader", router(web=web), sources=("web",), web_api_key="secret-key-value"
        )
        [error] = evidence.errors
        assert "401" in error
        assert "secret-key-value" not in error


class TestSharedDeadline:
    """One budget of wall-clock time across every source.

    Without it the slowest source sets the latency of the whole check: arXiv's
    3.5s spacing plus retry backoff alone could run past a minute.
    """

    async def test_a_source_that_never_answers_is_cut_off(self):
        import time

        never = asyncio.Event()

        async def npm(request):
            await never.wait()

        def github(request):
            hit = ("acme/nwd-reader", "Reads NWD files", "2026-09-01T00:00:00Z")
            return httpx.Response(200, json=repos([hit]))

        started = time.monotonic()
        evidence = await collect(
            ["nwd reader"], router(npm=npm, github=github), sources=("github", "npm"), deadline=0.3
        )
        assert time.monotonic() - started < 5
        assert [item.title for item in evidence.items] == ["acme/nwd-reader"]
        [error] = evidence.errors
        assert error.startswith("npm")
        assert "deadline" in error

    async def test_results_found_before_the_deadline_are_kept(self):
        never = asyncio.Event()

        async def github(request):
            if query_of(request, "q").startswith("nwd reader"):
                hit = ("acme/nwd-reader", "Reads NWD files", "2026-09-01T00:00:00Z")
                return httpx.Response(200, json=repos([hit]))
            await never.wait()

        evidence = await collect(
            ["nwd reader", "glb writer"], router(github=github), sources=("github",), deadline=0.3
        )
        assert [item.title for item in evidence.items] == ["acme/nwd-reader"]
        assert any("deadline" in error and "kept 1" in error for error in evidence.errors)


class TestBoundedConcurrency:
    """Sources run concurrently, but never more than a fixed number of requests
    at once -- across all of them, the arXiv fallback included."""

    async def _peak(self, limit: int, *, arxiv_refuses: bool = False) -> int:
        import time

        loop = asyncio.get_running_loop()
        state = {"now": 0, "peak": 0}

        async def busy(work):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            try:
                await loop.run_in_executor(None, time.sleep, 0.02)
                return work()
            finally:
                state["now"] -= 1

        async def handler(request):
            source = HOSTS[request.url.host]
            if source == "arxiv" and arxiv_refuses:
                return await busy(lambda: httpx.Response(406, text=""))
            return await busy(lambda: empty(source))

        async def fallback(url: str) -> str:
            return await busy(lambda: atom([]))

        await collect(
            ["alpha bravo"], handler, sources=KEYLESS, max_concurrency=limit, arxiv_fallback=fallback
        )
        return state["peak"]

    async def test_requests_in_flight_never_exceed_the_limit(self):
        assert await self._peak(2) == 2

    async def test_a_limit_of_one_serialises_everything(self):
        assert await self._peak(1) == 1

    async def test_the_arxiv_fallback_counts_against_the_limit(self):
        assert await self._peak(1, arxiv_refuses=True) == 1


class TestSourceSelection:
    def test_the_default_sources_need_no_key(self):
        from sota_anchor.retriever import KEYLESS_SOURCES

        assert set(KEYLESS_SOURCES) == set(KEYLESS)

    async def test_every_default_source_is_queried(self):
        hit: set[str] = set()

        def track(request):
            hit.add(HOSTS[request.url.host])
            return empty(HOSTS[request.url.host])

        await collect("nwd reader", track)
        assert hit == set(KEYLESS)

    async def test_an_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="pypi"):
            await collect("nwd reader", router(), sources=("pypi",))

    async def test_every_failing_source_is_reported_once(self):
        def down(request):
            raise httpx.ConnectError("offline")

        evidence = await collect("nwd reader", down)
        assert evidence.is_empty
        assert sorted(error.split(":")[0] for error in evidence.errors) == sorted(KEYLESS)


class TestEvidenceRenderingPerSource:
    """A date means something different per source; say which."""

    @staticmethod
    def _render(source: str, detail: str = "") -> str:
        item = Evidence(
            source=source,
            title="t",
            url="https://example.com",
            published=dt.date(2026, 9, 1),
            snippet="s",
            detail=detail,
        )
        return EvidenceSet(items=[item]).render()

    def test_a_repository_date_is_its_last_push(self):
        assert "last push: 2026-09-01" in self._render("github")

    def test_a_package_date_is_its_latest_release(self):
        assert "latest release: 2026-09-01" in self._render("npm")
        assert "latest release: 2026-09-01" in self._render("crates")

    def test_a_paper_date_is_its_publication(self):
        assert "published: 2026-09-01" in self._render("arxiv")
        assert "published: 2026-09-01" in self._render("huggingface")

    def test_detail_is_shown_beside_the_date(self):
        assert "2026-09-01 (0 stars)" in self._render("github", "0 stars")


class TestEngineeringScenario:
    """The prompt a live session declined to check, run through retrieval.

    The fake GitHub below ANDs a query's terms over name and description, as the
    real one does, across repositories that exist: an independent NWD reader, a
    converter, and unrelated projects that share the acronym. The vectors are
    what the new inversion prompt asks for: the most specific term first.
    """

    CORPUS = (
        (
            "1817716374/nwd-reader",
            "Independent C++20 Navisworks NWD/NWC/NWF reader for geometry, properties, "
            "materials, textures and model hierarchy.",
            "2026-09-06T18:53:21Z",
        ),
        (
            "AnT1pal/NWD2DWG",
            "Converter of Autodesk Navisworks (.NWD/.NWC) 3D geometry to DWG.",
            "2026-09-01T00:00:00Z",
        ),
        (
            "jwwangchn/NWD",
            "Official code for a Normalized Gaussian Wasserstein Distance for tiny objects.",
            "2026-06-21T00:00:00Z",
        ),
        ("blockdiag/nwdiag", "Network diagram generator.", "2026-01-20T00:00:00Z"),
    )

    def _github(self, request):
        terms = query_of(request, "q").lower().split()[:-1]  # drop the pushed bound
        hits = [
            repo for repo in self.CORPUS if all(t in f"{repo[0]} {repo[1]}".lower() for t in terms)
        ]
        return httpx.Response(200, json=repos(hits))

    async def _titles(self) -> list[str]:
        evidence = await collect(
            ["nwd glb conversion", "nwd navisworks reader"], router(github=self._github)
        )
        return [item.title for item in evidence.items]

    async def test_the_independent_reader_is_found(self):
        assert "1817716374/nwd-reader" in await self._titles()

    async def test_acronym_collisions_are_not_evidence(self):
        titles = await self._titles()
        assert "jwwangchn/NWD" not in titles
        assert "blockdiag/nwdiag" not in titles


class TestWebKeyStaysWithBrave:
    """httpx drops Authorization on a cross-host redirect, but not Brave's own
    X-Subscription-Token header. The web request therefore follows no redirect,
    so the key can only ever reach the host it was meant for."""

    async def test_a_redirect_is_not_followed_with_the_key(self):
        hosts: list[str] = []
        routed = router(
            web=lambda request: httpx.Response(
                302, headers={"Location": "https://elsewhere.example/collect"}
            )
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            hosts.append(request.url.host)
            if request.url.host == "elsewhere.example":
                return empty("web")
            return await routed(request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            evidence = await gather_evidence(
                "nwd reader",
                client=client,
                now=NOW,
                sources=("web",),
                web_api_key="secret-key-value",
            )

        assert "elsewhere.example" not in hosts
        [error] = evidence.errors
        assert "302" in error
        assert "secret-key-value" not in error
