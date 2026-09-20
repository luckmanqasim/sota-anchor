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
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from typing import Literal, Sequence

import httpx
from pydantic import BaseModel, Field

ARXIV_URL = "https://export.arxiv.org/api/query"
GITHUB_URL = "https://api.github.com/search/repositories"

DEFAULT_WINDOW_MONTHS = 12
DEFAULT_PER_SOURCE = 5
ARXIV_FETCH_SIZE = 25
MIN_TERMS = 2
MIN_TERM_LENGTH = 3
GITHUB_MAX_TERMS = 4

#: arXiv asks for roughly one request every three seconds and enforces it: after a
#: burst it answers 406 with an empty body from its Fastly edge (and 429 to other
#: clients). An unthrottled relaxation ladder trips this on the first real query.
ARXIV_MIN_INTERVAL = 3.0
MAX_ARXIV_ATTEMPTS = 3
MAX_GITHUB_ATTEMPTS = 3
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5.0
RETRY_STATUSES = frozenset({406, 429, 500, 502, 503, 504})

ATOM = "{http://www.w3.org/2005/Atom}"

#: Generic English function words. Deliberately not a topic vocabulary - dropping
#: "the" and "from" is grammar, and it is what keeps the AND-join specific enough
#: to matter without encoding anything about a domain.
STOPWORDS = frozenset(
    """
    a an the and or but nor for yet so of in on at to from by with without into onto
    upon about above below over under between among through during before after since
    is are was were be been being am do does did doing done have has had having
    can cannot could shall should will would may might must not no nor only just
    that this these those there here it its they them their which who whom whose what
    when where why how all any both each few more most other some such than too very
    we our you your he she his her him me my i us
    using use used via as if then else also etc per
    """.split()
)

Source = Literal["arxiv", "github"]

USER_AGENT = "sota-anchor/0.1 (+https://github.com/sota-anchor/sota-anchor)"


def make_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """The HTTP client this module expects.

    A descriptive User-Agent matters: arXiv's terms ask API clients to identify
    themselves, and an unidentified burst is the first thing an edge throttles.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        http2=True,
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
    return [
        " AND ".join(f"all:{term}" for term in terms[:size])
        for size in range(len(terms), MIN_TERMS - 1, -1)
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
    text: str,
    *,
    client: httpx.AsyncClient,
    now: dt.datetime,
    months: int,
    per_source: int,
    sleep: Sleeper,
) -> list[Evidence]:
    ladder = build_arxiv_queries(text)[:MAX_ARXIV_ATTEMPTS]
    if not ladder:
        return []
    terms = extract_terms(text)
    cutoff = _shift_months(now, months).date()

    for rung, search_query in enumerate(ladder):
        if rung:
            # Space out ladder rungs rather than bursting: a throttled arXiv
            # returns nothing at all, which is indistinguishable from "no recent
            # work exists" and would quietly bias every verdict toward "not obsolete".
            await sleep(ARXIV_MIN_INTERVAL)
        response = await _get_with_retry(
            client,
            ARXIV_URL,
            params={
                "search_query": search_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": ARXIV_FETCH_SIZE,
            },
            sleep=sleep,
            source="arxiv",
        )
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            raise

        found: list[Evidence] = []
        for entry in root.findall(f"{ATOM}entry"):
            title = (entry.findtext(f"{ATOM}title") or "").strip()
            summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            url = (entry.findtext(f"{ATOM}id") or "").strip()
            published = _parse_date(entry.findtext(f"{ATOM}published"))
            if not title or published is None or published < cutoff:
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
            return found[:per_source]
    return []


async def _search_github(
    text: str,
    *,
    client: httpx.AsyncClient,
    now: dt.datetime,
    months: int,
    per_source: int,
    token: str | None,
    sleep: Sleeper,
) -> list[Evidence]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    terms = extract_terms(text)
    ladder = build_github_queries(text, now=now, months=months)[:MAX_GITHUB_ATTEMPTS]
    for query in ladder:
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

        found: list[Evidence] = []
        for repo in response.json().get("items") or []:
            description = (repo.get("description") or "").strip()
            if not description:
                continue
            published = _parse_date(repo.get("pushed_at"))
            if published is None:
                continue
            found.append(
                Evidence(
                    source="github",
                    title=str(repo.get("full_name") or ""),
                    url=str(repo.get("html_url") or ""),
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
            return found[:per_source]
    return []


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def gather_evidence(
    text: str,
    *,
    client: httpx.AsyncClient | None = None,
    now: dt.datetime | None = None,
    months: int = DEFAULT_WINDOW_MONTHS,
    per_source: int = DEFAULT_PER_SOURCE,
    github_token: str | None = None,
    sleep: Sleeper | None = None,
) -> EvidenceSet:
    """Query every source concurrently and merge what came back.

    A failing source is recorded in ``errors`` and the others still contribute:
    partial evidence is useful, and a total outage must be visible to the caller
    rather than silently looking like "nothing has changed".
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    owns_client = client is None
    client = client or make_client()
    sleep = sleep or asyncio.sleep

    try:
        results = await asyncio.gather(
            _search_arxiv(
                text,
                client=client,
                now=now,
                months=months,
                per_source=per_source,
                sleep=sleep,
            ),
            _search_github(
                text,
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
    for name, result in zip(("arxiv", "github"), results):
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
    terms = extract_terms(text)
    evidence.items.sort(
        key=lambda item: (
            -_overlap(f"{item.title} {item.snippet}", terms),
            -item.published.toordinal(),
        )
    )
    return evidence
