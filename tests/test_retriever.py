"""Temporal-gated evidence retrieval.

The regression guarded here is the one that matters most: `all:{query}` with a
multi-word query is parsed by arXiv as an implicit OR across every term. Sorted
by submittedDate that returns the newest papers on arXiv about *anything*
(329,590 matches for the spec's own example), so the arbiter would judge every
proposal against unrelated noise while appearing to work.
"""

from __future__ import annotations

import datetime as dt
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

    def test_three_letter_acronyms_survive(self):
        assert "all:vlm" in build_arxiv_queries(QUERY)[0]

    def test_ladder_starts_most_specific(self):
        ladder = build_arxiv_queries(QUERY)
        assert len(ladder[0].split(" AND ")) > len(ladder[-1].split(" AND "))

    def test_ladder_drops_one_term_per_rung(self):
        ladder = build_arxiv_queries("alpha bravo charlie delta")
        assert [len(q.split(" AND ")) for q in ladder] == [4, 3, 2]

    def test_ladder_drops_the_least_salient_term_first(self):
        # Shortest term goes first: 'vlm' before any longer term.
        second = build_arxiv_queries(QUERY)[1]
        assert "all:vlm" not in second
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
                QUERY, client=client, now=NOW, months=12, **kwargs
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
            return await gather_evidence(QUERY, client=client, now=NOW, months=12, **kwargs)

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

        await self._gather(handler, github_token="ghp_example")
        assert seen.get("authorization") == "Bearer ghp_example"

    async def test_no_auth_header_without_a_token(self):
        seen: dict[str, str] = {}

        def handler(request):
            if GITHUB_URL in str(request.url):
                seen.update(request.headers)
                return httpx.Response(200, json=repos([]))
            return httpx.Response(200, text=atom([]))

        await self._gather(handler, github_token=None)
        assert "authorization" not in seen


class TestDegradation:
    async def _gather(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await gather_evidence(QUERY, client=client, now=NOW, months=12, **kwargs)

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
    def test_default_client_negotiates_http2(self):
        from sota_anchor.retriever import make_client

        client = make_client()
        try:
            assert client._transport._pool._http2 is True
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
            await gather_evidence(QUERY, client=client, now=NOW, months=12, sleep=fake_sleep)

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
            await gather_evidence(QUERY, client=client, now=NOW, months=12, sleep=fake_sleep)

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
                return httpx.Response(406, text="")
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
            await gather_evidence(QUERY, client=client, now=NOW, months=12, sleep=fake_sleep)

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

    def test_most_salient_term_survives_every_rung(self):
        from sota_anchor.retriever import build_github_queries

        for query in build_github_queries(QUERY, now=NOW, months=12):
            assert "multimodal" in query

    async def test_relaxes_when_the_narrow_query_finds_nothing(self):
        attempts: list[str] = []

        def handler(request):
            if GITHUB_URL not in str(request.url):
                return httpx.Response(200, text=atom([]))
            attempts.append(parse_qs(urlparse(str(request.url)).query)["q"][0])
            if len(attempts) < 3:
                return httpx.Response(200, json=repos([]))
            return httpx.Response(
                200,
                json=repos([("acme/found", "Polygon extraction.", "2026-09-18T09:00:00Z")]),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            evidence = await gather_evidence(QUERY, client=client, now=NOW, months=12)

        assert len(attempts) == 3
        assert len(attempts[0].split()) > len(attempts[2].split())
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
