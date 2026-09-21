"""Channel 2b - temporal-gated delta ingestion.

Two sources, queried concurrently, both filtered to a rolling recency window.
Nothing here knows about any discipline: the only word knowledge is a list of
generic English function words, which is linguistic rather than topical, so a
query about compiler IRs is built exactly like one about MEP penetrations.
"""

from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, Field

from . import __version__

ARXIV_URL = "https://export.arxiv.org/api/query"
GITHUB_URL = "https://api.github.com/search/repositories"

DEFAULT_WINDOW_MONTHS = 12
DEFAULT_PER_SOURCE = 5
ARXIV_FETCH_SIZE = 25
MIN_TERMS = 2
MIN_TERM_LENGTH = 3
GITHUB_MAX_TERMS = 4

#: arXiv's Terms of Use ask for "no more than one request every three seconds,
#: and limit requests to a single connection at a time". 3.5s leaves margin.
#: Both halves are enforced: exceed the rate and it answers 406 with an empty
#: body, and reuse a keep-alive connection and it answers 406 as well.
ARXIV_MIN_INTERVAL = 3.5
#: Widest arXiv query to start from. A six-term AND is effectively guaranteed
#: empty; a four-term AND returned the one on-target paper for a real proposal.
#: Starting wider spends the most expensive attempt on a query that cannot match.
ARXIV_MAX_TERMS = 4

#: Per-vector attempt allowance, and the ceiling across all vectors. A flat
#: shared budget was too shallow: three attempts split across two vectors let
#: neither relax past its first rung.
MAX_ARXIV_ATTEMPTS = 3
ARXIV_TOTAL_CAP = 6
MAX_GITHUB_ATTEMPTS = 3
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5.0
RETRY_STATUSES = frozenset({406, 429, 500, 502, 503, 504})

ATOM = "{http://www.w3.org/2005/Atom}"

#: Generic English function words. Deliberately not a topic vocabulary - dropping
#: "the" and "from" is grammar, and it is what keeps the AND-join specific enough
#: to matter without encoding anything about a domain.
STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so", "of", "in", "on", "at", "to", "from", "by", "with", "without", "into", "onto", "upon", "about", "above", "below", "over", "under", "between", "among", "through", "during", "before", "after", "since", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "done", "have", "has", "had", "having", "can", "cannot", "could", "shall", "should", "will", "would", "may", "might", "must", "not", "no", "nor", "only", "just", "that", "this", "these", "those", "there", "here", "it", "its", "they", "them", "their", "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "than", "too", "very", "we", "our", "you", "your", "he", "she", "his", "her", "him", "me", "my", "i", "us", "using", "use", "used", "via", "as", "if", "then", "else", "also", "etc", "per"]
)

Source = Literal["arxiv", "github"]

USER_AGENT = f"sota-anchor/{__version__} (+https://github.com/luckmanqasim/sota-anchor)"

#: arXiv rejects a request made on a reused connection. Sent per arXiv request;
#: under HTTP/2 this header is illegal, which is why the client stays on 1.1.
CLOSE_CONNECTION = {"Connection": "close"}


def make_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """The HTTP client this module expects.

    Keep-alive is disabled deliberately. arXiv answers `406` with an empty body
    to a request issued on a reused connection: measured against the live
    endpoint with everything else held identical, a shared keep-alive client
    got 406 while a fresh connection per request got 200 every time. That is
    their Terms of Use asking clients to "limit requests to a single connection
    at a time", enforced.

    The cost is one TCP handshake per request, which is irrelevant next to the
    3.5s spacing between them. HTTP/2 is off because it multiplexes onto one
    connection, which is the behaviour being avoided, and because `Connection:
    close` is not a legal HTTP/2 header.

    The User-Agent names a contact because it is good citizenship. It is not the
    fix: arXiv's Terms of Use state no User-Agent requirement, and an academic
    `mailto:` agent was measured still returning 406 over a reused connection.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        http2=False,
        limits=httpx.Limits(max_keepalive_connections=0),
        headers={"User-Agent": USER_AGENT},
    )


class Throttled(Exception):
    """A source asked us to slow down rather than refusing outright."""

    def __init__(self, source: str, status: int):
        self.source = source
        self.status = status
        super().__init__(
            f"throttled (HTTP {status}); this source limits how often it can be queried, "
            "so its evidence was skipped for this run"
        )


class Evidence(BaseModel):
    source: Source
    title: str
    url: str
    published: dt.date
    snippet: str


class EvidenceSet(BaseModel):
    items: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    window_months: int = DEFAULT_WINDOW_MONTHS

    @property
    def is_empty(self) -> bool:
        return not self.items

    def render(self) -> str:
        """A single evidence block, dated and attributed.

        Dates are explicit so the judge can weigh recency itself rather than
        taking "recent" on trust.
        """
        if self.is_empty:
            return "No evidence retrieved."
        lines = [f"Evidence from the past {self.window_months} months:", ""]
        for index, item in enumerate(self.items, start=1):
            lines += [
                f"{index}. [{item.source}] {_asciify(item.title)}",
                f"   published: {item.published.isoformat()}",
                f"   url: {item.url}",
                f"   {_asciify(item.snippet)}",
                "",
            ]
        return "\n".join(lines).rstrip() + "\n"


def _asciify(text: str) -> str:
    """Flatten to ASCII so output survives a cp1252 console."""
    # Escapes rather than literals: this table exists to match these exact
    # code points, and a literal smart quote in source is easy to mangle.
    folded = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")


def _shift_months(moment: dt.datetime, months: int) -> dt.datetime:
    total = (moment.year * 12 + moment.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def extract_terms(text: str) -> list[str]:
    """Salient terms, most salient first.

    Salience is length, with first appearance breaking ties, so the relaxation
    ladder sheds the vaguest term first and keeps the distinctive ones longest.
    """
    seen: dict[str, int] = {}
    for position, token in enumerate(re.findall(r"[A-Za-z0-9_+#.]+", text.lower())):
        token = token.strip(".")
        if len(token) < MIN_TERM_LENGTH or token in STOPWORDS or token in seen:
            continue
        seen[token] = position
    return sorted(seen, key=lambda term: (-len(term), seen[term]))


def build_arxiv_queries(text: str) -> list[str]:
    """A relaxation ladder of AND-joined queries, strictest first.

    ``search_query=all:{phrase}`` is parsed by arXiv as an implicit OR over every
    word, which combined with a recency sort returns the newest papers about
    anything at all. Each term therefore gets its own ``all:`` prefix and the
    terms are explicitly AND-joined. Because AND can over-constrain, callers walk
    the ladder until a rung returns results.
    """
    terms = extract_terms(text)
    if not terms:
        return []
    if len(terms) == 1:
        return [f"all:{terms[0]}"]
    widest = min(len(terms), ARXIV_MAX_TERMS)
    return [
        " AND ".join(f"all:{term}" for term in terms[:size])
        for size in range(widest, MIN_TERMS - 1, -1)
    ]


def build_github_query(
    text: str, *, now: dt.datetime, months: int, max_terms: int = GITHUB_MAX_TERMS
) -> str:
    """Terms plus a ``pushed:>`` bound computed from ``now``, never a fixed date."""
    since = _shift_months(now, months).date().isoformat()
    terms = extract_terms(text)[:max_terms]
    return " ".join([*terms, f"pushed:>{since}"])


def build_github_queries(text: str, *, now: dt.datetime, months: int) -> list[str]:
    """A relaxation ladder for GitHub, narrowest first.

    GitHub ANDs its terms as well, and a repository description is an order of
    magnitude shorter than an abstract, so a query specific enough for arXiv
    matches no repository at all. Measured on the spec's own example: six terms
    returned 0 results, four returned 0, three returned 0, two returned 41.
    Without a ladder this channel is silently empty forever.
    """
    available = len(extract_terms(text))
    if not available:
        return []
    widest = min(available, GITHUB_MAX_TERMS)
    return [
        build_github_query(text, now=now, months=months, max_terms=size)
        for size in range(widest, 0, -1)
    ]


def _overlap(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


Sleeper = Callable[[float], Awaitable[None]]
ArxivFallback = Callable[[str], Awaitable[str]]


def arxiv_url(params: Sequence[tuple[str, object]]) -> str:
    """Build an arXiv query URL, percent-encoding spaces.

    Never `+`: arXiv reads it as a literal plus inside `search_query`, which
    makes the field expression invalid rather than merely unmatched.
    """
    return f"{ARXIV_URL}?{urlencode(list(params), quote_via=quote)}"


async def fetch_arxiv_via_urllib(url: str, timeout: float = 30.0) -> str:
    """Fetch through the standard library, off the event loop.

    arXiv's edge answers httpx with 406 where it answers curl and urllib with
    200. Measured repeatedly on a clean request quota, varying headers, header
    order, percent-encoding, HTTP version, redirect handling and keep-alive:
    httpx is refused and urllib is not, and the discriminator was never found.
    Rather than let the academic channel go dark -- GitHub descriptions cannot
    demonstrate a capability leap on their own -- a refused request retries here.
    """

    def _read() -> str:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")

    return await asyncio.to_thread(_read)


#: Resolved at call time rather than bound as a default argument, so a test
#: suite can substitute it and stay offline.
DEFAULT_ARXIV_FALLBACK: ArxivFallback = fetch_arxiv_via_urllib


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, object],
    sleep: Sleeper,
    source: str,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET with backoff on the statuses that mean "slow down", not "no".

    Exhausted retries raise :class:`Throttled` rather than an
    ``HTTPStatusError``, whose message is a wall of percent-encoded URL that
    tells a user nothing about what to do.
    """
    for attempt in range(MAX_RETRIES + 1):
        response = await client.get(url, params=params, headers=headers)
        if response.status_code in RETRY_STATUSES:
            if attempt < MAX_RETRIES:
                await sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            raise Throttled(source, response.status_code)
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")  # pragma: no cover


async def _search_arxiv(
    vectors: Sequence[str],
    *,
    client: httpx.AsyncClient,
    now: dt.datetime,
    months: int,
    per_source: int,
    sleep: Sleeper,
    fallback: ArxivFallback | None = None,
) -> list[Evidence]:
    """Walk each query vector's relaxation ladder until the shared budget runs out.

    The budget is shared rather than per-vector: against a host that throttles
    this aggressively, three vectors must not mean three full ladders. Vectors
    are walked strictly in sequence, never concurrently, because arXiv asks for
    one connection at a time.
    """
    cutoff = _shift_months(now, months).date()
    ladders = [
        (vector, build_arxiv_queries(vector))
        for vector in vectors
        if build_arxiv_queries(vector)
    ]
    if not ladders:
        return []

    # Interleave rungs so every vector gets its strictest query tried before any
    # vector's looser ones -- a broad capability query is worth more at rung 0
    # than a narrow domain query is at rung 2.
    rungs: list[tuple[str, str]] = []
    for depth in range(max(len(ladder) for _, ladder in ladders)):
        for vector, ladder in ladders:
            if depth < len(ladder):
                rungs.append((vector, ladder[depth]))

    budget = min(MAX_ARXIV_ATTEMPTS * len(ladders), ARXIV_TOTAL_CAP)
    collected: list[Evidence] = []
    seen: set[str] = set()
    satisfied: set[str] = set()
    attempts = 0

    for vector, search_query in rungs:
        if attempts >= budget or len(collected) >= per_source:
            break
        # A vector that has already produced results does not need its looser
        # rungs; relaxation exists to rescue a vector that found nothing.
        if vector in satisfied:
            continue
        if attempts:
            await sleep(ARXIV_MIN_INTERVAL)
        attempts += 1

        query_params = [
            ("search_query", search_query),
            ("sortBy", "submittedDate"),
            ("sortOrder", "descending"),
            ("max_results", ARXIV_FETCH_SIZE),
        ]
        try:
            response = await _get_with_retry(
                client,
                ARXIV_URL,
                params=dict(query_params),
                sleep=sleep,
                source="arxiv",
                headers=dict(CLOSE_CONNECTION),
            )
            payload = response.text
        except Throttled:
            if fallback is None:
                raise
            try:
                payload = await fallback(arxiv_url(query_params))
            except Exception as error:
                raise Throttled("arxiv", 406) from error

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            raise

        terms = extract_terms(vector)
        found: list[Evidence] = []
        for entry in root.findall(f"{ATOM}entry"):
            title = (entry.findtext(f"{ATOM}title") or "").strip()
            summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            url = (entry.findtext(f"{ATOM}id") or "").strip()
            published = _parse_date(entry.findtext(f"{ATOM}published"))
            if not title or published is None or published < cutoff:
                continue
            if url in seen:
                continue
            found.append(
                Evidence(
                    source="arxiv",
                    title=title,
                    url=url,
                    published=published,
                    snippet=summary[:600],
                )
            )
        if found:
            found.sort(
                key=lambda item: (
                    -_overlap(f"{item.title} {item.snippet}", terms),
                    -item.published.toordinal(),
                )
            )
            satisfied.add(vector)
            for item in found[: per_source - len(collected)]:
                seen.add(item.url)
                collected.append(item)

    return collected


async def _search_github(
    vectors: Sequence[str],
    *,
    client: httpx.AsyncClient,
    now: dt.datetime,
    months: int,
    per_source: int,
    token: str | None,
    sleep: Sleeper,
) -> list[Evidence]:
    """Same ladder and shared budget as arXiv, against repository descriptions."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    ladders = [
        (vector, build_github_queries(vector, now=now, months=months))
        for vector in vectors
        if build_github_queries(vector, now=now, months=months)
    ]
    if not ladders:
        return []

    rungs: list[tuple[str, str]] = []
    for depth in range(max(len(ladder) for _, ladder in ladders)):
        for vector, ladder in ladders:
            if depth < len(ladder):
                rungs.append((vector, ladder[depth]))

    collected: list[Evidence] = []
    seen: set[str] = set()
    satisfied: set[str] = set()
    attempts = 0

    for vector, query in rungs:
        if attempts >= MAX_GITHUB_ATTEMPTS or len(collected) >= per_source:
            break
        if vector in satisfied:
            continue
        attempts += 1

        response = await _get_with_retry(
            client,
            GITHUB_URL,
            params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_source * 2,
            },
            headers=headers,
            sleep=sleep,
            source="github",
        )

        terms = extract_terms(vector)
        found: list[Evidence] = []
        for repo in response.json().get("items") or []:
            description = (repo.get("description") or "").strip()
            if not description:
                continue
            published = _parse_date(repo.get("pushed_at"))
            if published is None:
                continue
            url = str(repo.get("html_url") or "")
            if url in seen:
                continue
            found.append(
                Evidence(
                    source="github",
                    title=str(repo.get("full_name") or ""),
                    url=url,
                    published=published,
                    snippet=description[:400],
                )
            )
        if found:
            found.sort(
                key=lambda item: (
                    -_overlap(f"{item.title} {item.snippet}", terms),
                    -item.published.toordinal(),
                )
            )
            satisfied.add(vector)
            for item in found[: per_source - len(collected)]:
                seen.add(item.url)
                collected.append(item)

    return collected


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def gather_evidence(
    text: str | Sequence[str],
    *,
    client: httpx.AsyncClient | None = None,
    now: dt.datetime | None = None,
    months: int = DEFAULT_WINDOW_MONTHS,
    per_source: int = DEFAULT_PER_SOURCE,
    github_token: str | None = None,
    sleep: Sleeper | None = None,
    arxiv_fallback: ArxivFallback | None = None,
) -> EvidenceSet:
    """Query every source concurrently and merge what came back.

    A failing source is recorded in ``errors`` and the others still contribute:
    partial evidence is useful, and a total outage must be visible to the caller
    rather than silently looking like "nothing has changed".
    """
    now = now or dt.datetime.now(dt.UTC)
    owns_client = client is None
    client = client or make_client()
    sleep = sleep or asyncio.sleep

    # One query or a ladder of vectors; a single string is the common case.
    arxiv_fallback = arxiv_fallback or DEFAULT_ARXIV_FALLBACK
    vectors = [text] if isinstance(text, str) else [v for v in text if v and v.strip()]
    if not vectors:
        return EvidenceSet(window_months=months)

    try:
        results = await asyncio.gather(
            _search_arxiv(
                vectors,
                client=client,
                now=now,
                months=months,
                per_source=per_source,
                sleep=sleep,
                fallback=arxiv_fallback,
            ),
            _search_github(
                vectors,
                client=client,
                now=now,
                months=months,
                per_source=per_source,
                token=github_token,
                sleep=sleep,
            ),
            return_exceptions=True,
        )
    finally:
        if owns_client:
            await client.aclose()

    evidence = EvidenceSet(window_months=months)
    for name, result in zip(("arxiv", "github"), results, strict=True):
        if isinstance(result, Throttled):
            evidence.errors.append(f"{name}: {result}")
        elif isinstance(result, BaseException):
            evidence.errors.append(f"{name}: {type(result).__name__}: {result}")
        else:
            evidence.items.extend(result)

    # Relevance leads, recency breaks ties. Sorting the merged list by date alone
    # would let a barely-related paper from last week outrank the one that
    # actually addresses the question, and the top items carry the most weight in
    # the judge prompt. Everything here already passed the recency window.
    terms = [term for vector in vectors for term in extract_terms(vector)]
    evidence.items.sort(
        key=lambda item: (
            -_overlap(f"{item.title} {item.snippet}", terms),
            -item.published.toordinal(),
        )
    )
    return evidence
