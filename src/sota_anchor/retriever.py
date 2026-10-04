"""Channel 2b - temporal-gated evidence retrieval across papers, code and packages.

Every source is filtered to a rolling recency window and queried concurrently:

* arXiv - papers, through a keyword AND ladder;
* Hugging Face Papers - the same literature through semantic search, which finds
  work that arXiv's keyword matching misses;
* GitHub - repositories;
* npm and crates.io - published packages;
* the web, through Brave Search - only when ``SOTA_ANCHOR_BRAVE_API_KEY`` is set, so the
  default install needs no key at all.

PyPI is absent by necessity rather than choice: it offers no search API, and its
search page answers clients with a JavaScript challenge.

Nothing here knows about any discipline. The only word knowledge is a list of
generic English function words, which is linguistic rather than topical, so a
query about NWD readers is built exactly like one about compiler IRs.

Three budgets hold across every source: a per-source attempt budget, spent rung
by rung down a relaxation ladder; one wall-clock deadline, after which whatever
has been found is kept and the rest reported; and a cap on how many requests
are in flight at once.
"""

from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import html
import os
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlencode

import httpx
from pydantic import BaseModel, Field

from . import __version__

ARXIV_URL = "https://export.arxiv.org/api/query"
GITHUB_URL = "https://api.github.com/search/repositories"
HF_PAPERS_URL = "https://huggingface.co/api/papers/search"
NPM_SEARCH_URL = "https://registry.npmjs.org/-/v1/search"
CRATES_URL = "https://crates.io/api/v1/crates"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

#: Switches the web source on. A variable of this project's own: a key already set
#: for another tool stays that tool's, and the plugin hands this one over from its
#: settings.
WEB_API_KEY_ENV = "SOTA_ANCHOR_BRAVE_API_KEY"

#: Optional; raises GitHub's unauthenticated search limit of ten a minute. Named
#: for this project for the same reason: a token set for other tools stays theirs.
GITHUB_AUTH_ENVS = ("SOTA_ANCHOR_GITHUB_TOKEN",)

DEFAULT_WINDOW_MONTHS = 12
#: The widest window the MCP tool accepts. Past ten years the search dates stop
#: meaning "recent", and far enough past it they stop being valid dates at all.
MAX_WINDOW_MONTHS = 120
DEFAULT_PER_SOURCE = 5
ARXIV_FETCH_SIZE = 25
HF_FETCH_SIZE = 20
REGISTRY_FETCH_SIZE = 20
WEB_FETCH_SIZE = 10
MIN_TERMS = 2
MIN_TERM_LENGTH = 3

#: Widest GitHub query to start from. Repository descriptions are an order of
#: magnitude shorter than abstracts: measured on the spec's own example, a
#: four-term AND matched no repository and a two-term AND matched 41.
GITHUB_MAX_TERMS = 3
#: Package registries rank fuzzily rather than ANDing, so width matters less;
#: three terms keeps the first request specific.
REGISTRY_MAX_TERMS = 3

#: arXiv's Terms of Use ask for "no more than one request every three seconds,
#: and limit requests to a single connection at a time". 3.5s leaves margin.
#: Both halves are enforced: exceed the rate and it answers 406 with an empty
#: body, and reuse a keep-alive connection and it answers 406 as well.
ARXIV_MIN_INTERVAL = 3.5
#: Widest arXiv query to start from. A six-term AND is effectively guaranteed
#: empty; a four-term AND returned the one on-target paper for a real proposal.
#: Starting wider spends the most expensive attempt on a query that cannot match.
ARXIV_MAX_TERMS = 4
#: crates.io's crawler policy: at most one request per second.
CRATES_MIN_INTERVAL = 1.0

#: Per-vector attempt allowances, and the ceiling across all vectors. A flat
#: shared budget was too shallow: three attempts split across two vectors let
#: neither relax past its first rung -- measured on arXiv, and then again on
#: GitHub, where it stopped both vectors short of the two-term rung that finds
#: repositories.
MAX_ARXIV_ATTEMPTS = 3
ARXIV_TOTAL_CAP = 6
MAX_GITHUB_ATTEMPTS = 2
GITHUB_TOTAL_CAP = 4

#: How many of its vector's terms a result must mention to count as evidence
#: (all of them, for a vector shorter than this). Semantic and fuzzy search
#: never come back empty -- Hugging Face answered "navisworks nwd reader" with
#: marine-fog nowcasting -- and a one-word GitHub rung matches every project
#: that happens to share an acronym.
MIN_OVERLAP = 2

#: Wall-clock budget for one retrieval, shared by every source. Past it, what
#: each source has found is kept and the shortfall is reported.
DEFAULT_DEADLINE_SECONDS = 60.0
#: Requests in flight at once, across every source.
DEFAULT_MAX_CONCURRENCY = 4

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

Source = Literal["arxiv", "github", "huggingface", "npm", "crates", "web"]

#: Queried by default: none of these needs an account.
KEYLESS_SOURCES: tuple[Source, ...] = ("arxiv", "github", "huggingface", "npm", "crates")
#: In merge order. arXiv precedes Hugging Face, so a paper both return keeps
#: arXiv's full abstract.
ALL_SOURCES: tuple[Source, ...] = (*KEYLESS_SOURCES, "web")

#: What an item's date means depends on where it came from. The judge weighs
#: recency itself, so it has to know which recency it is looking at.
DATE_LABELS: dict[str, str] = {
    "arxiv": "published",
    "huggingface": "published",
    "github": "last push",
    "npm": "latest release",
    "crates": "latest release",
    "web": "page date",
}

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

    The User-Agent names the tool because it is good citizenship, and because
    crates.io's crawler policy requires it. It is not the arXiv fix: arXiv's
    Terms of Use state no User-Agent requirement, and an academic `mailto:`
    agent was measured still returning 406 over a reused connection.
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
    #: Signals for weighing maturity rather than mere existence: stars, the
    #: latest version, downloads, upvotes, a linked implementation.
    detail: str = ""


class EvidenceSet(BaseModel):
    items: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    window_months: int = DEFAULT_WINDOW_MONTHS

    @property
    def is_empty(self) -> bool:
        return not self.items

    def render(self) -> str:
        """A single evidence block, dated and attributed.

        Dates are explicit, and labelled with what they mean for their source,
        so the judge can weigh recency itself rather than taking "recent" on
        trust.
        """
        if self.is_empty:
            return "No evidence retrieved."
        lines = [f"Evidence from the past {self.window_months} months:", ""]
        for index, item in enumerate(self.items, start=1):
            dated = f"{DATE_LABELS.get(item.source, 'date')}: {item.published.isoformat()}"
            if item.detail:
                dated += f" ({_one_line(item.detail)})"
            lines += [
                f"{index}. [{item.source}] {_one_line(item.title)}",
                f"   {dated}",
                f"   url: {_one_line(item.url)}",
                f"   {_one_line(item.snippet)}",
                "",
            ]
        return "\n".join(lines).rstrip() + "\n"


def _asciify(text: str) -> str:
    """Flatten to ASCII so output survives a cp1252 console."""
    # Escapes rather than literals: this table exists to match these exact
    # code points, and a literal smart quote in source is easy to mangle.
    folded = (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")


def _one_line(text: str) -> str:
    """ASCII on a single line. Every field here was written by whoever published
    the result, and a line break inside one would let it pass for a line of the
    prompt around it."""
    return " ".join(_asciify(text).split())


def _plain(value: object) -> str:
    """Markup-free, whitespace-collapsed text: search snippets arrive as HTML."""
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return " ".join(text.split())


def _shift_months(moment: dt.datetime, months: int) -> dt.datetime:
    total = (moment.year * 12 + moment.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def extract_terms(text: str) -> list[str]:
    """Search terms, in the order the query's author wrote them.

    Order is salience: every ladder relaxes by dropping terms from the end. The
    author knows which term is specific -- the inversion prompts ask for it
    first -- and no measure computed here does. Length was tried, and it shed
    the short identifiers first: `nwd`, `glb`, `mep`, the terms that make an
    engineering query specific at all.
    """
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_+#.]+", text.lower()):
        token = token.strip(".")
        if len(token) < MIN_TERM_LENGTH or token in STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms


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


def build_term_ladder(text: str, *, max_terms: int) -> list[str]:
    """Space-joined queries, widest first, shedding the last-written term per rung."""
    terms = extract_terms(text)
    widest = min(len(terms), max_terms)
    return [" ".join(terms[:size]) for size in range(widest, 0, -1)]


def _overlap(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _admissible(text: str, terms: Sequence[str], floor: int) -> bool:
    """Whether a result mentions enough of its vector's terms to count."""
    return _overlap(text, terms) >= min(floor, len(terms))


Sleeper = Callable[[float], Awaitable[None]]
ArxivFallback = Callable[[str], Awaitable[str]]


class _Getter(Protocol):
    async def get(self, url: str, **kwargs: Any) -> httpx.Response: ...


class _BoundedClient:
    """The client every source shares, admitting a fixed number of requests at once.

    The limit is held only for the request itself, never across the politeness
    sleeps between requests, so a source waiting out arXiv's spacing does not
    starve the others.
    """

    def __init__(self, client: httpx.AsyncClient, limit: asyncio.Semaphore) -> None:
        self._client = client
        self._limit = limit

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        async with self._limit:
            return await self._client.get(url, **kwargs)


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
    client: _Getter,
    url: str,
    *,
    params: dict[str, object],
    sleep: Sleeper,
    source: str,
    headers: dict[str, str] | None = None,
    refusals: frozenset[int] = frozenset(),
    follow_redirects: bool = True,
) -> httpx.Response:
    """GET with backoff on the statuses that mean "slow down", not "no".

    Exhausted retries raise :class:`Throttled` rather than an
    ``HTTPStatusError``, whose message is a wall of percent-encoded URL that
    tells a user nothing about what to do. A status in ``refusals`` raises it at
    once: it is known not to yield to a retry, so backing off only spends the
    shared deadline.
    """
    for attempt in range(MAX_RETRIES + 1):
        response = await client.get(
            url, params=params, headers=headers, follow_redirects=follow_redirects
        )
        if response.status_code in refusals:
            raise Throttled(source, response.status_code)
        if response.status_code in RETRY_STATUSES:
            if attempt < MAX_RETRIES:
                await sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            raise Throttled(source, response.status_code)
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_date(value: object) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


# --- the ladder walker -------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What one source may spend on one retrieval."""

    per_vector: int
    total_cap: int
    #: Seconds between this source's own requests.
    spacing: float = 0.0
    #: Relevance floor; 0 leaves relevance to the source's own query semantics.
    min_overlap: int = MIN_OVERLAP

    def attempts(self, vectors: int) -> int:
        return min(self.per_vector * vectors, self.total_cap)


BUDGETS: dict[str, Budget] = {
    # arXiv ANDs every term already and matches stemmed forms that a substring
    # count cannot see, so a floor there would reject on-target papers.
    "arxiv": Budget(MAX_ARXIV_ATTEMPTS, ARXIV_TOTAL_CAP, ARXIV_MIN_INTERVAL, min_overlap=0),
    "github": Budget(MAX_GITHUB_ATTEMPTS, GITHUB_TOTAL_CAP),
    # Semantic search takes a phrase whole; there is nothing to relax.
    "huggingface": Budget(1, 3, spacing=0.5),
    "npm": Budget(2, 4, spacing=0.5),
    "crates": Budget(2, 4, spacing=CRATES_MIN_INTERVAL),
    "web": Budget(1, 3, spacing=1.1),
}


@dataclass(frozen=True)
class _Candidate:
    """A parsed result, with the fuller text its relevance is judged on.

    The displayed snippet is truncated; the text a result matched on is not.
    """

    item: Evidence
    text: str


RungFetcher = Callable[[str], Awaitable[list[_Candidate]]]
Ladders = Sequence[tuple[str, Sequence[str]]]


def _interleave(ladders: Ladders) -> list[tuple[str, str]]:
    """Every vector's strictest rung before any vector's looser ones.

    A broad capability query is worth more at rung 0 than a narrow domain query
    is at rung 2.
    """
    rungs: list[tuple[str, str]] = []
    for depth in range(max(len(ladder) for _, ladder in ladders)):
        for vector, ladder in ladders:
            if depth < len(ladder):
                rungs.append((vector, ladder[depth]))
    return rungs


async def _walk(
    ladders: Ladders,
    fetch_rung: RungFetcher,
    *,
    budget: Budget,
    per_source: int,
    sink: list[Evidence],
    sleep: Sleeper,
) -> None:
    """Walk each vector's relaxation ladder until the source's budget runs out.

    A vector that has produced admissible results stops relaxing: relaxation
    exists to rescue a vector that found nothing, and a rung whose results all
    fall below the relevance floor has not rescued it. Results land in ``sink``
    as they are found, so a deadline that cuts the walk short keeps them.
    """
    ladders = [(vector, ladder) for vector, ladder in ladders if ladder]
    if not ladders:
        return

    allowed = budget.attempts(len(ladders))
    seen = {item.url for item in sink}
    satisfied: set[str] = set()
    attempts = 0

    for vector, query in _interleave(ladders):
        if attempts >= allowed or len(sink) >= per_source:
            break
        if vector in satisfied:
            continue
        if attempts and budget.spacing:
            await sleep(budget.spacing)
        attempts += 1

        terms = extract_terms(vector)
        found = [
            candidate
            for candidate in await fetch_rung(query)
            if candidate.item.url not in seen
            and _admissible(candidate.text, terms, budget.min_overlap)
        ]
        if not found:
            continue

        # Relevance leads, recency breaks ties.
        found.sort(
            key=lambda c: (-_overlap(c.text, terms), -c.item.published.toordinal())
        )
        satisfied.add(vector)
        for candidate in found:
            if len(sink) >= per_source:
                break
            if candidate.item.url in seen:
                continue
            seen.add(candidate.item.url)
            sink.append(candidate.item)


@dataclass(frozen=True)
class _Context:
    """What every source search shares within one retrieval."""

    client: _Getter
    now: dt.datetime
    months: int
    per_source: int
    sleep: Sleeper

    @property
    def cutoff(self) -> dt.date:
        return _shift_months(self.now, self.months).date()

    async def walk(
        self, source: str, ladders: Ladders, fetch_rung: RungFetcher, sink: list[Evidence]
    ) -> None:
        await _walk(
            ladders,
            fetch_rung,
            budget=BUDGETS[source],
            per_source=self.per_source,
            sink=sink,
            sleep=self.sleep,
        )


# --- sources -----------------------------------------------------------------


def _parse_arxiv(payload: str, cutoff: dt.date) -> list[_Candidate]:
    root = ET.fromstring(payload)
    candidates: list[_Candidate] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        url = (entry.findtext(f"{ATOM}id") or "").strip()
        published = _parse_date(entry.findtext(f"{ATOM}published"))
        if not title or published is None or published < cutoff:
            continue
        item = Evidence(
            source="arxiv", title=title, url=url, published=published, snippet=summary[:600]
        )
        candidates.append(_Candidate(item, f"{title} {summary}"))
    return candidates


async def _search_arxiv(
    vectors: Sequence[str],
    ctx: _Context,
    sink: list[Evidence],
    *,
    fallback: ArxivFallback | None,
) -> None:
    """Strictly sequential: arXiv asks for one connection at a time.

    A 406 goes straight to the fallback. arXiv's edge refuses httpx with 406
    however often it is retried, and measured live the backoff before each
    fallback cost 45 seconds of a 55-second retrieval. A 429 or 5xx is still
    backed off, since those do yield to waiting.
    """
    refusals = frozenset({406}) if fallback is not None else frozenset()

    async def rung(search_query: str) -> list[_Candidate]:
        query_params = [
            ("search_query", search_query),
            ("sortBy", "submittedDate"),
            ("sortOrder", "descending"),
            ("max_results", ARXIV_FETCH_SIZE),
        ]
        try:
            response = await _get_with_retry(
                ctx.client,
                ARXIV_URL,
                params=dict(query_params),
                sleep=ctx.sleep,
                source="arxiv",
                headers=dict(CLOSE_CONNECTION),
                refusals=refusals,
            )
            payload = response.text
        except Throttled:
            if fallback is None:
                raise
            try:
                payload = await fallback(arxiv_url(query_params))
            except Exception as error:
                raise Throttled("arxiv", 406) from error
        return _parse_arxiv(payload, ctx.cutoff)

    ladders = [(vector, build_arxiv_queries(vector)) for vector in vectors]
    await ctx.walk("arxiv", ladders, rung, sink)


def _parse_github(payload: Any) -> list[_Candidate]:
    # No window check here: the query's own pushed:> bound applies it.
    candidates: list[_Candidate] = []
    for repo in payload.get("items") or []:
        description = " ".join(str(repo.get("description") or "").split())
        published = _parse_date(repo.get("pushed_at"))
        if not description or published is None:
            continue
        name = str(repo.get("full_name") or "")
        stars = repo.get("stargazers_count")
        item = Evidence(
            source="github",
            title=name,
            url=str(repo.get("html_url") or ""),
            published=published,
            snippet=description[:400],
            detail=f"{stars} stars" if isinstance(stars, int) else "",
        )
        topics = " ".join(str(topic) for topic in repo.get("topics") or [])
        candidates.append(_Candidate(item, f"{name} {description} {topics}"))
    return candidates


async def _search_github(
    vectors: Sequence[str], ctx: _Context, sink: list[Evidence], *, token: str | None
) -> None:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def rung(query: str) -> list[_Candidate]:
        response = await _get_with_retry(
            ctx.client,
            GITHUB_URL,
            params={"q": query, "sort": "updated", "order": "desc", "per_page": ctx.per_source * 2},
            headers=headers,
            sleep=ctx.sleep,
            source="github",
        )
        return _parse_github(response.json())

    ladders = [
        (vector, build_github_queries(vector, now=ctx.now, months=ctx.months))
        for vector in vectors
    ]
    await ctx.walk("github", ladders, rung, sink)


def _parse_huggingface(payload: Any, cutoff: dt.date) -> list[_Candidate]:
    if not isinstance(payload, list):
        raise ValueError("unexpected response shape from Hugging Face paper search")
    candidates: list[_Candidate] = []
    for result in payload:
        if not isinstance(result, dict):
            continue
        paper = result.get("paper") if isinstance(result.get("paper"), dict) else result
        arxiv_id = str(paper.get("id") or "").strip()
        title = " ".join(str(paper.get("title") or result.get("title") or "").split())
        summary = " ".join(str(paper.get("summary") or result.get("summary") or "").split())
        published = _parse_date(paper.get("publishedAt") or result.get("publishedAt"))
        if not arxiv_id or not title or published is None or published < cutoff:
            continue
        signals = []
        if isinstance(paper.get("upvotes"), int):
            signals.append(f"{paper['upvotes']} upvotes")
        if paper.get("githubRepo"):
            signals.append(f"code: {paper['githubRepo']}")
        item = Evidence(
            source="huggingface",
            title=title,
            url=f"https://huggingface.co/papers/{arxiv_id}",
            published=published,
            snippet=summary[:600],
            detail="; ".join(signals),
        )
        keywords = " ".join(str(keyword) for keyword in paper.get("ai_keywords") or [])
        candidates.append(_Candidate(item, f"{title} {summary} {keywords}"))
    return candidates


async def _search_huggingface(vectors: Sequence[str], ctx: _Context, sink: list[Evidence]) -> None:
    """Semantic search over papers, taking each vector as a whole phrase."""

    async def rung(query: str) -> list[_Candidate]:
        response = await _get_with_retry(
            ctx.client,
            HF_PAPERS_URL,
            params={"q": query, "limit": HF_FETCH_SIZE},
            sleep=ctx.sleep,
            source="huggingface",
        )
        return _parse_huggingface(response.json(), ctx.cutoff)

    ladders = [(vector, [vector.strip()]) for vector in vectors if extract_terms(vector)]
    await ctx.walk("huggingface", ladders, rung, sink)


def _parse_npm(payload: Any, cutoff: dt.date) -> list[_Candidate]:
    if not isinstance(payload, dict):
        raise ValueError("unexpected response shape from the npm registry")
    candidates: list[_Candidate] = []
    for entry in payload.get("objects") or []:
        package = entry.get("package") if isinstance(entry, dict) else None
        if not isinstance(package, dict):
            continue
        name = str(package.get("name") or "").strip()
        description = " ".join(str(package.get("description") or "").split())
        published = _parse_date(package.get("date"))
        if not name or not description or published is None or published < cutoff:
            continue
        links = package.get("links") or {}
        version = package.get("version")
        item = Evidence(
            source="npm",
            title=name,
            url=str(links.get("npm") or f"https://www.npmjs.com/package/{name}"),
            published=published,
            snippet=description[:400],
            detail=f"version {version}" if version else "",
        )
        keywords = " ".join(str(keyword) for keyword in package.get("keywords") or [])
        candidates.append(_Candidate(item, f"{name} {description} {keywords}"))
    return candidates


async def _search_npm(vectors: Sequence[str], ctx: _Context, sink: list[Evidence]) -> None:
    async def rung(query: str) -> list[_Candidate]:
        response = await _get_with_retry(
            ctx.client,
            NPM_SEARCH_URL,
            params={"text": query, "size": REGISTRY_FETCH_SIZE},
            sleep=ctx.sleep,
            source="npm",
        )
        return _parse_npm(response.json(), ctx.cutoff)

    ladders = [(vector, build_term_ladder(vector, max_terms=REGISTRY_MAX_TERMS)) for vector in vectors]
    await ctx.walk("npm", ladders, rung, sink)


def _parse_crates(payload: Any, cutoff: dt.date) -> list[_Candidate]:
    if not isinstance(payload, dict):
        raise ValueError("unexpected response shape from crates.io")
    candidates: list[_Candidate] = []
    for crate in payload.get("crates") or []:
        if not isinstance(crate, dict):
            continue
        name = str(crate.get("name") or "").strip()
        description = " ".join(str(crate.get("description") or "").split())
        published = _parse_date(crate.get("updated_at"))
        if not name or not description or published is None or published < cutoff:
            continue
        version = (
            crate.get("max_stable_version") or crate.get("max_version") or crate.get("newest_version")
        )
        downloads = crate.get("downloads")
        signals = [
            f"version {version}" if version else "",
            f"{downloads:,} downloads" if isinstance(downloads, int) else "",
        ]
        item = Evidence(
            source="crates",
            title=name,
            url=f"https://crates.io/crates/{name}",
            published=published,
            snippet=description[:400],
            detail="; ".join(signal for signal in signals if signal),
        )
        keywords = " ".join(str(keyword) for keyword in crate.get("keywords") or [])
        candidates.append(_Candidate(item, f"{name} {description} {keywords}"))
    return candidates


async def _search_crates(vectors: Sequence[str], ctx: _Context, sink: list[Evidence]) -> None:
    """crates.io requires an identifying User-Agent, so it is sent per request
    rather than trusted to whichever client the caller supplied."""

    async def rung(query: str) -> list[_Candidate]:
        response = await _get_with_retry(
            ctx.client,
            CRATES_URL,
            params={"q": query, "per_page": REGISTRY_FETCH_SIZE},
            headers={"User-Agent": USER_AGENT},
            sleep=ctx.sleep,
            source="crates",
        )
        return _parse_crates(response.json(), ctx.cutoff)

    ladders = [(vector, build_term_ladder(vector, max_terms=REGISTRY_MAX_TERMS)) for vector in vectors]
    await ctx.walk("crates", ladders, rung, sink)


def _parse_web(payload: Any, cutoff: dt.date) -> list[_Candidate]:
    if not isinstance(payload, dict):
        raise ValueError("unexpected response shape from Brave web search")
    candidates: list[_Candidate] = []
    for result in (payload.get("web") or {}).get("results") or []:
        if not isinstance(result, dict):
            continue
        title = _plain(result.get("title"))
        url = str(result.get("url") or "").strip()
        description = _plain(result.get("description"))
        # An undated page cannot be weighed for recency, so it is not evidence
        # here -- even inside a date-filtered search.
        published = _parse_date(result.get("page_age"))
        if not title or not url or published is None or published < cutoff:
            continue
        item = Evidence(
            source="web", title=title, url=url, published=published, snippet=description[:400]
        )
        extra = " ".join(_plain(snippet) for snippet in result.get("extra_snippets") or [])
        candidates.append(_Candidate(item, f"{title} {description} {extra} {url}"))
    return candidates


async def _search_web(
    vectors: Sequence[str], ctx: _Context, sink: list[Evidence], *, api_key: str
) -> None:
    """General web search. The key travels in a header, never in the URL, so no
    error message or log line can echo it."""
    freshness = f"{ctx.cutoff.isoformat()}to{ctx.now.date().isoformat()}"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

    async def rung(query: str) -> list[_Candidate]:
        response = await _get_with_retry(
            ctx.client,
            BRAVE_URL,
            params={"q": query, "count": WEB_FETCH_SIZE, "freshness": freshness},
            headers=headers,
            sleep=ctx.sleep,
            source="web",
            # httpx strips Authorization on a cross-host redirect, but not this
            # header, so a redirect is reported rather than followed with the key.
            follow_redirects=False,
        )
        return _parse_web(response.json(), ctx.cutoff)

    ladders = [(vector, [vector.strip()]) for vector in vectors if extract_terms(vector)]
    await ctx.walk("web", ladders, rung, sink)


# --- orchestration -----------------------------------------------------------

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")


def _identity(item: Evidence) -> str:
    """The key under which two sources' copies of one paper collapse to one."""
    if item.source in ("arxiv", "huggingface"):
        match = _ARXIV_ID.search(item.url.rstrip("/"))
        if match:
            return f"arxiv:{match.group(1)}"
    return item.url


def _choose_sources(
    requested: Sequence[str] | None, *, web_key: bool
) -> tuple[list[str], list[str]]:
    """The sources to query, in merge order, plus any notes on sources skipped."""
    if requested is None:
        return [*KEYLESS_SOURCES, *(["web"] if web_key else [])], []

    unknown = [name for name in requested if name not in ALL_SOURCES]
    if unknown:
        raise ValueError(
            f"unknown evidence source: {', '.join(unknown)}; "
            f"choose from {', '.join(ALL_SOURCES)}"
        )
    chosen = [name for name in ALL_SOURCES if name in requested]
    notes: list[str] = []
    if "web" in chosen and not web_key:
        chosen.remove("web")
        notes.append(f"web: skipped; set {WEB_API_KEY_ENV} to enable general web search")
    return chosen, notes


async def _run_source(
    name: str,
    search: Callable[[list[Evidence]], Awaitable[None]],
    *,
    deadline_at: float,
    deadline: float,
) -> tuple[list[Evidence], str | None]:
    """Run one source under the shared deadline, turning every failure into a note.

    A failing source is recorded and the others still contribute: partial
    evidence is useful, and a total outage must be visible to the caller rather
    than silently looking like "nothing has changed".
    """
    sink: list[Evidence] = []
    timer = asyncio.timeout_at(deadline_at)
    try:
        async with timer:
            await search(sink)
    except TimeoutError as error:
        if not timer.expired():
            return sink, f"{name}: {type(error).__name__}: {error}"
        kept = f"; kept {len(sink)} result{'' if len(sink) == 1 else 's'}" if sink else ""
        return sink, f"{name}: stopped at the {deadline:g}s retrieval deadline{kept}"
    except Throttled as error:
        return sink, f"{name}: {error}"
    except httpx.HTTPStatusError as error:
        response = error.response
        return sink, f"{name}: HTTP {response.status_code} {response.reason_phrase}".rstrip()
    except Exception as error:
        return sink, f"{name}: {type(error).__name__}: {error}"
    return sink, None


def _from_environment(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        # The plugin fills these from its settings as ${user_config.KEY}. An option
        # left unfilled must never be sent anywhere as if it were a key.
        if value and not value.startswith("${"):
            return value
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
    sources: Sequence[str] | None = None,
    web_api_key: str | None = None,
    deadline: float = DEFAULT_DEADLINE_SECONDS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> EvidenceSet:
    """Query every chosen source concurrently and merge what came back.

    ``sources`` defaults to every keyless source, plus the web when a key is
    available. Each source walks its own relaxation ladder within its own attempt
    budget; all of them share one deadline and one limit on requests in flight.
    """
    web_api_key = web_api_key or _from_environment([WEB_API_KEY_ENV])
    github_token = github_token or _from_environment(GITHUB_AUTH_ENVS)
    chosen, notes = _choose_sources(sources, web_key=bool(web_api_key))

    # One query or a ladder of vectors; a single string is the common case.
    vectors = [text] if isinstance(text, str) else [v for v in text if v and v.strip()]
    if not vectors:
        return EvidenceSet(window_months=months)

    now = now or dt.datetime.now(dt.UTC)
    sleep = sleep or asyncio.sleep
    arxiv_fallback = arxiv_fallback or DEFAULT_ARXIV_FALLBACK
    owns_client = client is None
    client = client or make_client()

    limit = asyncio.Semaphore(max_concurrency)

    async def bounded_fallback(url: str) -> str:
        async with limit:
            return await arxiv_fallback(url)

    ctx = _Context(
        client=_BoundedClient(client, limit),
        now=now,
        months=months,
        per_source=per_source,
        sleep=sleep,
    )
    searches: dict[str, Callable[[list[Evidence]], Awaitable[None]]] = {
        "arxiv": lambda sink: _search_arxiv(vectors, ctx, sink, fallback=bounded_fallback),
        "github": lambda sink: _search_github(vectors, ctx, sink, token=github_token),
        "huggingface": lambda sink: _search_huggingface(vectors, ctx, sink),
        "npm": lambda sink: _search_npm(vectors, ctx, sink),
        "crates": lambda sink: _search_crates(vectors, ctx, sink),
        "web": lambda sink: _search_web(vectors, ctx, sink, api_key=web_api_key or ""),
    }

    deadline_at = asyncio.get_running_loop().time() + deadline
    try:
        outcomes = await asyncio.gather(
            *(
                _run_source(name, searches[name], deadline_at=deadline_at, deadline=deadline)
                for name in chosen
            )
        )
    finally:
        if owns_client:
            await client.aclose()

    evidence = EvidenceSet(window_months=months, errors=list(notes))
    identities: set[str] = set()
    for found, error in outcomes:
        if error:
            evidence.errors.append(error)
        for item in found:
            identity = _identity(item)
            if identity in identities:
                continue
            identities.add(identity)
            evidence.items.append(item)

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
