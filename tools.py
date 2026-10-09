"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

ATOM = "{http://www.w3.org/2005/Atom}"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SUMMARY_CHARS = 600
FETCH_CHARS = 12000
ARXIV_MIN_INTERVAL = 3.0
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
HEADERS = {"User-Agent": "deep-research-lab/1.0 (educational project)"}


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0, sleep=time.sleep):
    """Call fn(); when it raises RetryableError, wait and call it again (at most `attempts` calls in total).

    Wait = the server's Retry-After when given, else base * 2**attempt plus random jitter; always capped at `cap`.
    The last failure is re-raised without sleeping. Any other exception propagates immediately.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                backoff = base * 2 ** attempt
                delay = min(backoff + random.uniform(0, backoff), cap)
            sleep(max(delay, 0.0))
    raise ValueError("attempts must be >= 1")


def _retry_after(response):
    """Seconds from a Retry-After header (numeric form only), else None."""
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _get(url, params):
    """One GET; turns transient failures into RetryableError and other HTTP errors into exceptions."""
    try:
        response = httpx.get(url, params=params, headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", _retry_after(response))
    response.raise_for_status()
    return response


def _error(exc, secret=None):
    """'ERROR: <type>: <message>' with `secret` redacted."""
    message = f"ERROR: {type(exc).__name__}: {exc}"
    return message.replace(secret, "***") if secret else message


def _clean(text):
    return " ".join(str(text or "").split())


def _short(text, limit=SUMMARY_CHARS):
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clamp(value, low, high):
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return low


def _dumps(records, tool_name):
    for record in records:
        _remember(record["url"], tool_name, record["title"])
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- registry of every URL a research tool really returned (host memory, one research run per process) ----
# The agent must only cite these: a URL written from the model's memory (e.g. a wrong arXiv id) is not in it.
# Each URL remembers WHICH tool returned it (so `source` labels can be checked) and, for paper tools, the real title.
# web_fetch registers nothing: fetching a URL the model made up must not make it look verified.
_seen_urls = {}  # normalized url -> {"tools": {tool names}, "title": title from a paper tool or None}
_seen_lock = threading.Lock()


def _normalize_url(url):
    url = str(url or "").strip().rstrip("/.,;")
    url = re.sub(r"^http://", "https://", url, flags=re.I)
    return re.sub(r"^https://www\.", "https://", url, flags=re.I)


def _remember(url, tool_name, title=None):
    with _seen_lock:
        entry = _seen_urls.setdefault(_normalize_url(url), {"tools": set(), "title": None})
        entry["tools"].add(tool_name)
        entry["title"] = title or entry["title"]


def unverified_sources(sources):
    """Entries of sources.json whose url was never returned by a source tool in this process."""
    with _seen_lock:
        seen = set(_seen_urls)
    return [entry for entry in sources if _normalize_url(entry.get("url")) not in seen]


def tools_for(url):
    """Names of the tools that returned this url in this process (empty set if none)."""
    with _seen_lock:
        return set(_seen_urls.get(_normalize_url(url), {}).get("tools", ()))


def tool_title(url):
    """The title a paper tool returned for this url, or None (web pages, unknown urls)."""
    with _seen_lock:
        return _seen_urls.get(_normalize_url(url), {}).get("title")


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()   # researchers run in parallel: serialize arXiv calls to keep 3 s between them
_arxiv_last_call = 0.0


def _arxiv_get(params):
    global _arxiv_last_call
    with _arxiv_lock:
        wait = _arxiv_last_call + ARXIV_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return _get(ARXIV_URL, params)
        finally:
            _arxiv_last_call = time.monotonic()


def _arxiv_id(raw_id):
    """'http://arxiv.org/abs/2501.00001v2' -> '2501.00001' (also old-style ids such as 'hep-th/9901001v1')."""
    return re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1].strip())


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few keywords (e.g. "world model video"), newest submissions first.
    Use short keyword queries (2-5 words); all words must appear in the paper. Returns a JSON list of
    {id, url, published, title, summary} where url is https://arxiv.org/abs/<id>. Cite these with source "arxiv"."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query or "")
    terms = [t for t in terms if t.upper() not in {"AND", "OR", "ANDNOT", "NOT", "ALL", "TI", "ABS"}]
    if not terms:
        return "NO RESULTS"
    params = {"search_query": " AND ".join(f"all:{t}" for t in terms[:8]), "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": _clamp(max_results, 1, 30), "start": 0}
    try:
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        root = ET.fromstring(response.content)
        records = []
        for entry in root.findall(f"{ATOM}entry"):
            paper_id = _arxiv_id(entry.findtext(f"{ATOM}id", ""))
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": entry.findtext(f"{ATOM}published", "")[:10],
                            "title": _clean(entry.findtext(f"{ATOM}title")),
                            "summary": _short(entry.findtext(f"{ATOM}summary"))})
        return _dumps(records, "arxiv_search")
    except Exception as exc:  # noqa: BLE001 - a tool never raises
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_records(items):
    """Map Hugging Face paper items to compact records; skip items without paper.id."""
    records = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({"id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
                        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                        "title": _clean(paper.get("title") or item.get("title")),
                        "summary": _short(paper.get("ai_summary") or paper.get("summary") or item.get("summary")),
                        "upvotes": paper.get("upvotes") or 0,
                        "github": paper.get("githubRepo") or "",
                        "stars": paper.get("githubStars") or 0})
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic).
    Cite these with source "hf-daily"."""
    params = {"limit": _clamp(limit, 1, 100)}
    if date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()):
        params["date"] = date.strip()
    try:
        records = _hf_records(with_retry(lambda: _get(HF_DAILY_URL, params)).json())
        if keyword and keyword.strip():
            needle = keyword.strip().lower()
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return _dumps(records, "hf_daily_papers")
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic (semantic search, e.g. "world models for robotics"). Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} where url is https://huggingface.co/papers/<id>.
    Cite these with source "hf-search"."""
    if not (query or "").strip():
        return "NO RESULTS"
    params = {"q": query.strip()[:200], "limit": _clamp(limit, 1, 50)}
    try:
        return _dumps(_hf_records(with_retry(lambda: _get(HF_SEARCH_URL, params)).json()), "hf_search_papers")
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
_RATE_LIMIT_TEXT = re.compile(r"rate[ _-]?limit", re.I)


def _exa_rate_limited(result, text):
    """The free tier answers HTTP 200 and flags the limit in result._meta (plus a short notice as the text)."""
    meta = result.get("_meta") or {}
    if any("rate" in key.lower() and value for key, value in meta.items()):
        return True
    if not _RATE_LIMIT_TEXT.search(text):
        return False
    return bool(result.get("isError")) or (len(text) < 600 and "exa" in text.lower())


def _exa_call(name, arguments, register=False):
    """Call one Exa MCP tool (JSON-RPC over HTTP, SSE answer) and return its text. Retries rate limits."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {**HEADERS, "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def once():
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=httpx.Timeout(90.0, connect=10.0))
        except httpx.TransportError as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code in RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {response.status_code} from Exa", _retry_after(response))
        response.raise_for_status()
        message = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                message = json.loads(line[len("data:"):].strip())
        if message is None:  # plain JSON answer (no SSE framing)
            message = response.json()
        if message.get("error"):
            error = message["error"]
            raise RuntimeError(f"Exa JSON-RPC error {error.get('code')}: {error.get('message')}")
        result = message.get("result") or {}
        text = "\n".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa rate limit reached (free tier); set EXA_API_KEY to avoid it")
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {_short(text, 300)}")
        if register:  # only search results count as 'returned by a tool'; a fetched page proves nothing
            for found in re.findall(r"(?m)^URL:\s*(\S+)", text):
                _remember(found, name)
        return text.strip()

    try:
        return with_retry(once, attempts=6, base=5.0, cap=60.0), None
    except Exception as exc:  # noqa: BLE001
        return None, _error(exc, secret=key or None)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa) for blogs, surveys, project pages, docs and news. `query` = a natural-language
    description of the ideal page; `objective` = what you want to learn from it. Returns clean text of the top
    results, each with Title, URL, Published date and highlights. Cite these with source "web" (even arXiv pages)."""
    if not (query or "").strip():
        return "NO RESULTS"
    arguments = {"query": query.strip(), "objective": (objective or "").strip() or f"Find information about: {query}",
                 "numResults": _clamp(num_results, 1, 10)}
    text, error = _exa_call("web_search_exa", arguments, register=True)
    return error or text or "NO RESULTS"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page or a blog post) as clean text/markdown.
    Use it to verify a claim or to get details beyond a search snippet. Long pages are truncated to ~12000 chars."""
    if not str(url or "").startswith(("http://", "https://")):
        return "ERROR: ValueError: url must start with http:// or https://"
    text, error = _exa_call("web_fetch_exa", {"urls": [url]})
    if error:
        return error
    if not text:
        return "NO RESULTS"
    return text if len(text) <= FETCH_CHARS else text[:FETCH_CHARS] + "\n...[truncated]"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
