"""check_citations.py - Citation validator.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")  # [3]  [1, 2]  [1-3]; not a Markdown link [3](url)
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]")
_URL = re.compile(r"https?://[^\s<>()\[\]]+")


def _numbers(group):
    """'1, 2' -> [1, 2];  '1-3' -> [1, 2, 3]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def cited_numbers(body):
    """Every citation number in the body, ignoring code blocks/spans and Markdown links."""
    return {n for match in _CITATION.finditer(_CODE.sub(" ", body)) for n in _numbers(match.group(1))}


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems = []
    by_n, seen_urls = {}, {}
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources.json entry #{i} is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"sources.json entry #{i}: n={n!r} is not an integer")
            continue
        if n in by_n:
            problems.append(f"source number [{n}] appears twice in sources.json")
        by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}]: url {url!r} is not an http(s) URL")
        elif url in seen_urls:
            problems.append(f"source [{n}]: url {url} duplicates source [{seen_urls[url]}]")
        else:
            seen_urls[url] = n

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        return problems + ["report has no '## References' heading"]
    body, references = report_text[:headings[-1].start()], report_text[headings[-1].end():]

    cited = cited_numbers(body)
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys())]
    problems += [f"source [{n}] never cited in the report body" for n in sorted(by_n.keys() - cited)]

    ref_lines = {}
    for line in references.splitlines():
        match = _REF_LINE.match(line)
        if not match:
            continue
        n = int(match.group(1))
        if n in ref_lines:
            problems.append(f"reference [{n}] is listed more than once")
            continue
        ref_lines[n] = line
        if n not in by_n:
            problems.append(f"reference [{n}] is not a source in sources.json")
            continue
        urls = [u.rstrip(".,;:") for u in _URL.findall(line)]
        if len(urls) != 1:
            problems.append(f"reference [{n}] must hold exactly one URL, found {len(urls)}")
        elif urls[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] url {urls[0]} differs from sources.json url {by_n[n].get('url')}")
    problems += [f"source [{n}] has no line in ## References" for n in sorted(by_n.keys() - ref_lines.keys())]
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
