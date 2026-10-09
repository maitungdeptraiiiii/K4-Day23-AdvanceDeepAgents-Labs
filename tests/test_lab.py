"""Offline tests (no network, no LLM):   pip install pytest && pytest -q"""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research  # noqa: E402
import tools  # noqa: E402
from check_citations import check  # noqa: E402
from tools import RetryableError, with_retry  # noqa: E402


# ---- with_retry ----
def failing(times, retry_after=None):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= times:
            raise RetryableError("busy", retry_after)
        return "ok"
    return fn, calls


def test_retry_succeeds_after_failures_with_capped_backoff():
    sleeps = []
    fn, calls = failing(3)
    assert with_retry(fn, attempts=5, base=1.0, cap=3.0, sleep=sleeps.append) == "ok"
    assert len(calls) == 4 and len(sleeps) == 3
    assert 1.0 <= sleeps[0] <= 2.0 and all(s <= 3.0 for s in sleeps)


def test_retry_uses_retry_after_and_cap():
    sleeps = []
    fn, _ = failing(2, retry_after=7)
    with_retry(fn, attempts=3, cap=5.0, sleep=sleeps.append)
    assert sleeps == [5.0, 5.0]


def test_retry_gives_up_without_sleeping_after_last_attempt():
    sleeps = []
    fn, calls = failing(10)
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=3, sleep=sleeps.append)
    assert len(calls) == 3 and len(sleeps) == 2


def test_retry_does_not_retry_other_errors():
    calls = []

    def fn():
        calls.append(1)
        raise KeyError("bug")
    with pytest.raises(KeyError):
        with_retry(fn, sleep=lambda s: None)
    assert len(calls) == 1


# ---- tools (offline parts) ----
def test_arxiv_empty_query_never_calls_network(monkeypatch):
    monkeypatch.setattr(tools, "_arxiv_get", lambda params: pytest.fail("network called"))
    assert tools.arxiv_search.invoke({"query": "  :'\" AND OR "}) == "NO RESULTS"


def test_arxiv_parses_atom_and_strips_version(monkeypatch):
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v3</id>
    <published>2025-01-02T00:00:00Z</published><title>A  World\n Model</title><summary> Some
    text </summary></entry></feed>"""
    seen = {}

    def fake(params):
        seen.update(params)
        return httpx.Response(200, content=atom)
    monkeypatch.setattr(tools, "_arxiv_get", fake)
    records = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" model'}))
    assert seen["search_query"] == "all:world AND all:model"
    assert records == [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                        "title": "A World Model", "summary": "Some text"}]


def test_exa_rate_limit_is_detected_from_meta():
    assert tools._exa_rate_limited({"_meta": {"rateLimited": True}}, "anything")
    assert not tools._exa_rate_limited({}, "A long article that mentions nothing special.")


def test_exa_key_is_redacted(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    monkeypatch.setattr(tools.httpx, "post", lambda url, **kw: (_ for _ in ()).throw(ValueError(f"bad url {url}")))
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR:") and "secret-key-123" not in out


# ---- research.py ----
@pytest.mark.parametrize("topic,expected", [("Survey about World Model", "survey-about-world-model"),
                                            ("../../etc/passwd", "etc-passwd"), ("", "topic"), ("!!!", "topic")])
def test_slugify(topic, expected):
    assert research.slugify(topic) == expected


def test_slugify_truncates():
    assert len(research.slugify("word " * 40)) <= 60


def test_save_outputs_writes_nothing_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "download", lambda backend, paths: {p: None for p in paths})
    with pytest.raises(RuntimeError):
        research.save_outputs(None, "t", [], 1.0, "m", reports_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# ---- check_citations ----
SOURCES = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"},
           {"n": 2, "url": "https://huggingface.co/papers/2", "source": "hf-search"},
           {"n": 3, "url": "https://example.org/a", "source": "web"}]
REFS = ("## References\n[1] A. arxiv. https://arxiv.org/abs/1 (2025-01-01)\n"
        "[2] B. hf-search. https://huggingface.co/papers/2 (n.d.)\n[3] C. web. https://example.org/a (n.d.)\n")


def test_valid_report_with_grouped_citations():
    assert check("# T\nClaim [1, 2]. Other [3].\n\n" + REFS, SOURCES) == []
    assert check("# T\nClaim [1-3].\n\n" + REFS, SOURCES) == []


def test_missing_heading_and_empty_sources():
    assert check("text [1]", []) == ["no sources in sources.json"]
    assert any("References" in p for p in check("text [1][2][3]", SOURCES))


def test_uncited_and_unknown_numbers():
    problems = check("# T\n[1][2][4] `[3]` [3](https://x.org)\n\n" + REFS, SOURCES)
    assert "[4] cited but missing from sources.json" in problems
    assert "source [3] never cited in the report body" in problems


def test_bundled_and_wrong_reference_lines():
    refs = ("## References\n[1] A; B. https://arxiv.org/abs/1 https://arxiv.org/abs/9\n"
            "[2] B. https://huggingface.co/papers/22\n[2] dup https://huggingface.co/papers/2\n[7] x https://y.org\n")
    problems = check("[1][2][3]\n" + refs, SOURCES)
    assert any("[1] must hold exactly one URL" in p for p in problems)
    assert any("[2] url" in p for p in problems)
    assert any("[2] is listed more than once" in p for p in problems)
    assert any("[7] is not a source" in p for p in problems)
    assert "source [3] has no line in ## References" in problems


def test_bad_source_entries():
    bad = [{"n": "1", "url": "https://a.org"}, {"n": 2, "url": "ftp://b"}, {"n": 3, "url": "https://a.org"},
           {"n": 4, "url": "https://a.org"}]
    problems = check("[2][3][4]\n## References\n", bad)
    assert any("not an integer" in p for p in problems)
    assert any("not an http(s) URL" in p for p in problems)
    assert any("duplicates source [3]" in p for p in problems)


# ---- URL registry (fabricated sources are caught) ----
def test_unverified_sources_flags_urls_no_tool_returned(monkeypatch):
    monkeypatch.setattr(tools, "_seen_urls", {})
    tools._remember("https://arxiv.org/abs/1811.04551", "arxiv_search")
    sources = [{"n": 1, "url": "http://www.arxiv.org/abs/1811.04551/"}, {"n": 2, "url": "https://arxiv.org/abs/1811.08131"}]
    assert [e["n"] for e in tools.unverified_sources(sources)] == [2]


def test_web_search_parses_sse_and_registers_urls(monkeypatch):
    monkeypatch.setattr(tools, "_seen_urls", {})
    message = {"result": {"content": [{"type": "text", "text": "Title: A\nURL: https://e.org/a\n"}]}}
    body = f"event: message\ndata: {json.dumps(message)}\n"
    monkeypatch.setattr(tools.httpx, "post", lambda url, **kw: httpx.Response(200, text=body, request=httpx.Request("POST", url)))
    assert "https://e.org/a" in tools.web_search.invoke({"query": "x"})
    assert tools.unverified_sources([{"url": "https://e.org/a"}]) == []


def test_source_problems_checks_families_and_url_shape(monkeypatch):
    import agents
    urls = ["https://arxiv.org/abs/2501.00001", "https://example.org/x", "https://huggingface.co/papers/2502.00002"]
    monkeypatch.setattr(tools, "_seen_urls", {})
    tools._remember(urls[0], "arxiv_search")
    tools._remember(urls[1], "web_search_exa")
    tools._remember(urls[2], "hf_search_papers")
    ok = [{"n": 1, "url": urls[0], "source": "arxiv"}, {"n": 2, "url": urls[1], "source": "web"},
          {"n": 3, "url": urls[2], "source": "hf-search"}]
    assert agents.source_problems(ok) == []
    problems = agents.source_problems([{"n": 1, "url": urls[1], "source": "arxiv"}])
    assert any("needs a url of the form https://arxiv.org/abs/" in p for p in problems)
    assert any("only 1 source families" in p for p in problems)


def test_source_problems_flags_versioned_arxiv_url_and_wrong_title(monkeypatch):
    import agents
    monkeypatch.setattr(tools, "_seen_urls", {})
    tools._remember("https://arxiv.org/abs/2402.16363v5", "arxiv_search")
    tools._remember("https://huggingface.co/papers/2210.03350", "hf_search_papers", "Measuring the Compositionality Gap")
    problems = agents.source_problems([
        {"n": 1, "url": "https://arxiv.org/abs/2402.16363v5", "source": "arxiv", "title": "x"},
        {"n": 2, "url": "https://huggingface.co/papers/2210.03350", "source": "hf-search", "title": "Self-Ask with Search"}])
    assert any(p.startswith("[1]") and "no version suffix" in p for p in problems)
    assert any(p.startswith("[2] wrong title") for p in problems)


def test_label_must_match_the_tool_that_returned_the_url_and_fetch_registers_nothing(monkeypatch):
    import agents
    monkeypatch.setattr(tools, "_seen_urls", {})
    tools._remember("https://arxiv.org/abs/2501.00001", "web_search_exa")  # found by a web search
    problems = agents.source_problems([{"n": 1, "url": "https://arxiv.org/abs/2501.00001", "source": "arxiv"}])
    assert any("was returned by ['web_search_exa'], not by arxiv_search" in p for p in problems)

    # a page the model asked web_fetch for (a URL it made up) must not become "verified"
    page = "# Some other paper\nURL: https://arxiv.org/abs/2408.10379\n"
    message = {"result": {"content": [{"type": "text", "text": page}]}}
    monkeypatch.setattr(tools.httpx, "post", lambda url, **kw: httpx.Response(
        200, text=f"data: {json.dumps(message)}\n", request=httpx.Request("POST", url)))
    assert "Some other paper" in tools.web_fetch.invoke({"url": "https://arxiv.org/abs/2408.10379"})
    assert [e["n"] for e in tools.unverified_sources([{"n": 1, "url": "https://arxiv.org/abs/2408.10379"}])] == [1]
