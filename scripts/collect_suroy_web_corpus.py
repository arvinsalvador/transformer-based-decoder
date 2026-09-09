#!/usr/bin/env python3
"""
Collect a real/public SUROY web corpus from a curated source catalog.

Safety / methodology:
- Public HTTP(S) pages only; no login, cookies, private groups, or Facebook scraping.
- Honors robots.txt where available.
- Same-domain crawling only from curated seeds.
- Rate limited.
- Stores source URL and provenance for every extracted document.
- Exact normalized-text deduplication.
- Does not attempt to bypass blocks, CAPTCHAs, paywalls, or access controls.

Output:
  data/external/suroy_web/web_documents.jsonl
  data/external/suroy_web/source_results.csv
  data/external/suroy_web/collection_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

USER_AGENT = "Project-SUROY-Academic-Corpus/1.0 (+non-commercial university homework)"
REGION_TERMS = (
    "siargao", "surigao", "general luna", "general-luna", "dapa", "del carmen",
    "del-carmen", "pilar", "san isidro", "san-isidro", "santa monica",
    "santa-monica", "burgos", "san benito", "san-benito", "socorro",
    "bucas", "sohoton", "sugba", "cloud 9", "cloud-9", "magpupungko",
    "pacifico", "cagwait", "hinatuan", "britania", "enchanted river",
)
TOURISM_TERMS = (
    "tour", "tourism", "travel", "destination", "attraction", "beach", "surf",
    "island", "lagoon", "cave", "falls", "river", "mangrove", "itinerary",
    "hotel", "resort", "restaurant", "transport", "ferry", "airport",
    "snorkel", "diving", "kayak", "visitor", "guide", "things to do",
)
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4", ".mp3", ".zip",
    ".rar", ".7z", ".exe", ".dmg", ".apk", ".css", ".js", ".xml", ".rss",
)
BLOCKED_HOST_FRAGMENTS = ("facebook.com", "m.facebook.com", "instagram.com", "tiktok.com")

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()

def normalized_key(text: str) -> str:
    text = normalize_space(text).lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower().split(":")[0]

def canonical_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = re.sub(r"/+", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    # Strip tracking/fragment; preserve meaningful query parameters only for article URLs.
    query = parts.query if ("view=article" in parts.query or "id=" in parts.query) else ""
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))

def relevant_link(url: str, anchor: str) -> bool:
    low = (urllib.parse.unquote(url) + " " + anchor).lower()
    return any(t in low for t in REGION_TERMS) or (
        "surigaodelnorte.gov.ph" in low and any(t in low for t in TOURISM_TERMS)
    ) or (
        "surigaodelsur.gov.ph/tourism" in low
    ) or (
        "siargaofinder.com/blog" in low
    )

class Extractor(HTMLParser):
    """Conservative visible-text/link extractor without third-party dependencies."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.capture_depth = 0
        self.current = []
        self.blocks = []
        self.links = []
        self.title_parts = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg", "canvas", "form"):
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "blockquote", "td"):
            self.capture_depth += 1
            if self.capture_depth == 1:
                self.current = []
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append((href, ""))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg", "canvas", "form"):
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "blockquote", "td") and self.capture_depth:
            self.capture_depth -= 1
            if self.capture_depth == 0:
                text = normalize_space(" ".join(self.current))
                if text:
                    self.blocks.append(text)
                self.current = []

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = normalize_space(data)
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        if self.capture_depth:
            self.current.append(text)
        if self.links:
            href, anchor = self.links[-1]
            # Best-effort anchor text accumulation for the most recent link.
            self.links[-1] = (href, (anchor + " " + text).strip())

@dataclass
class RobotsCache:
    parsers: dict
    def __init__(self):
        self.parsers = {}

    def allowed(self, url: str) -> bool:
        h = host(url)
        if h not in self.parsers:
            robots_url = f"{urllib.parse.urlsplit(url).scheme}://{h}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                req = urllib.request.Request(robots_url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=10) as r:
                    body = r.read(512_000).decode("utf-8", "replace")
                rp.parse(body.splitlines())
            except Exception:
                # If robots.txt is unavailable, do not infer a prohibition.
                rp = None
            self.parsers[h] = rp
        rp = self.parsers[h]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

def fetch(url: str, timeout: int, max_bytes: int) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        ctype = response.headers.get_content_type()
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        return data, ctype

def quality_block(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    low = text.lower()
    bad = (
        "cookie policy", "privacy policy", "all rights reserved", "javascript is disabled",
        "subscribe to our newsletter", "accept cookies", "skip to content",
    )
    if any(x in low for x in bad) and len(text) < 500:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(1, len(text)) >= 0.45

def chunk_blocks(blocks: list[str], min_chars: int, target_chars: int, max_chars: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for block in blocks:
        block = normalize_space(block)
        if not quality_block(block, 40, max_chars):
            continue
        if cur and size + len(block) + 1 > target_chars:
            text = normalize_space(" ".join(cur))
            if len(text) >= min_chars:
                chunks.append(text[:max_chars])
            cur, size = [], 0
        cur.append(block)
        size += len(block) + 1
    if cur:
        text = normalize_space(" ".join(cur))
        if len(text) >= min_chars:
            chunks.append(text[:max_chars])
    return chunks

def extract_html(data: bytes, base_url: str, min_chars: int, target_chars: int, max_chars: int):
    text = data.decode("utf-8", "replace")
    parser = Extractor()
    parser.feed(text)
    title = normalize_space(" ".join(parser.title_parts))[:300]
    chunks = chunk_blocks(parser.blocks, min_chars, target_chars, max_chars)
    links = []
    for href, anchor in parser.links:
        try:
            absolute = canonical_url(urllib.parse.urljoin(base_url, href))
            if absolute.startswith(("http://", "https://")):
                links.append((absolute, normalize_space(anchor)[:300]))
        except Exception:
            continue
    return title, chunks, links

def extract_pdf(data: bytes, min_chars: int, target_chars: int, max_chars: int):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction") from exc
    doc = fitz.open(stream=data, filetype="pdf")
    blocks = []
    for page in doc:
        txt = normalize_space(page.get_text("text"))
        if txt:
            blocks.append(txt)
    return "", chunk_blocks(blocks, min_chars, target_chars, max_chars), []

def read_sources(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("allow_fetch", "1").strip() not in ("1", "true", "TRUE", "yes"):
                continue
            yield {
                **row,
                "crawl_depth": int(row.get("crawl_depth") or 0),
                "max_pages": int(row.get("max_pages") or 1),
            }

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="resources/suroy_web_sources.csv")
    p.add_argument("--output-dir", default="data/external/suroy_web")
    p.add_argument("--max-total-pages", type=int, default=300)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--delay", type=float, default=1.25)
    p.add_argument("--max-bytes", type=int, default=8_000_000)
    p.add_argument("--min-chars", type=int, default=250)
    p.add_argument("--target-chars", type=int, default=1400)
    p.add_argument("--max-chars", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    sources_path = Path(args.sources)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    docs_path = outdir / "web_documents.jsonl"
    results_path = outdir / "source_results.csv"
    manifest_path = outdir / "collection_manifest.json"
    if not args.overwrite and any(x.exists() for x in (docs_path, results_path, manifest_path)):
        print("Output exists. Use --overwrite only when intentionally rebuilding.", file=sys.stderr)
        return 2

    seeds = list(read_sources(sources_path))
    robots = RobotsCache()
    visited = set()
    seen_text = set()
    host_last_fetch = defaultdict(float)
    host_counts = defaultdict(int)
    result_rows = []
    total_pages = total_docs = failures = robots_skips = 0
    started = utc_now()

    queue = deque()
    seed_meta = {}
    for seed in seeds:
        u = canonical_url(seed["url"])
        queue.append((u, 0, seed["source_id"]))
        seed_meta[seed["source_id"]] = seed

    with docs_path.open("w", encoding="utf-8", newline="\n") as sink:
        while queue and total_pages < args.max_total_pages:
            url, depth, source_id = queue.popleft()
            url = canonical_url(url)
            if url in visited:
                continue
            visited.add(url)
            meta = seed_meta[source_id]
            h = host(url)
            if any(x in h for x in BLOCKED_HOST_FRAGMENTS):
                continue
            if urllib.parse.urlsplit(url).path.lower().endswith(SKIP_EXTENSIONS):
                continue
            if host_counts[h] >= meta["max_pages"]:
                continue
            if not robots.allowed(url):
                robots_skips += 1
                result_rows.append([source_id, url, "ROBOTS_SKIPPED", 0, 0, ""])
                continue

            wait = args.delay - (time.time() - host_last_fetch[h])
            if wait > 0:
                time.sleep(wait)
            try:
                data, ctype = fetch(url, args.timeout, args.max_bytes)
                host_last_fetch[h] = time.time()
                host_counts[h] += 1
                total_pages += 1
                if ctype == "application/pdf" or url.lower().endswith(".pdf"):
                    title, chunks, links = extract_pdf(
                        data, args.min_chars, args.target_chars, args.max_chars
                    )
                elif "html" in ctype or ctype in ("text/plain", "application/xhtml+xml"):
                    title, chunks, links = extract_html(
                        data, url, args.min_chars, args.target_chars, args.max_chars
                    )
                else:
                    result_rows.append([source_id, url, "UNSUPPORTED_TYPE", 0, len(data), ctype])
                    continue

                emitted = 0
                for idx, text in enumerate(chunks, 1):
                    key = normalized_key(text)
                    if key in seen_text:
                        continue
                    seen_text.add(key)
                    total_docs += 1
                    emitted += 1
                    record = {
                        "document_id": f"WEB-{total_docs:07d}",
                        "text": text,
                        "source_id": source_id,
                        "source_url": url,
                        "source_domain": h,
                        "source_title": title,
                        "source_type": meta["source_type"],
                        "region": meta["region"],
                        "category": meta["category"],
                        "provenance_type": "real_public_web",
                        "synthetic": False,
                        "retrieved_at": utc_now(),
                        "chunk_index": idx,
                        "text_sha256": key,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")

                result_rows.append([source_id, url, "SUCCESS", emitted, len(data), ctype])

                if depth < meta["crawl_depth"]:
                    for candidate, anchor in links:
                        if host(candidate) != h:
                            continue
                        if candidate in visited:
                            continue
                        if relevant_link(candidate, anchor):
                            queue.append((candidate, depth + 1, source_id))
            except urllib.error.HTTPError as exc:
                failures += 1
                result_rows.append([source_id, url, f"HTTP_{exc.code}", 0, 0, ""])
            except Exception as exc:
                failures += 1
                result_rows.append([source_id, url, f"FAILED_{type(exc).__name__}", 0, 0, str(exc)[:180]])

            if total_pages and total_pages % 10 == 0:
                print(f"Pages: {total_pages} | Documents: {total_docs} | Queue: {len(queue)}", flush=True)

    with results_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id","url","status","documents_emitted","bytes","detail"])
        w.writerows(result_rows)

    manifest = {
        "state": "completed",
        "started_at": started,
        "completed_at": utc_now(),
        "source_catalog": str(sources_path.resolve()),
        "seed_sources": len(seeds),
        "pages_fetched": total_pages,
        "documents_emitted": total_docs,
        "unique_text_hashes": len(seen_text),
        "fetch_failures": failures,
        "robots_skips": robots_skips,
        "max_total_pages": args.max_total_pages,
        "methodology": {
            "public_pages_only": True,
            "robots_respected": True,
            "facebook_automated_scraping": False,
            "same_domain_crawl": True,
            "rate_limit_seconds_per_host": args.delay,
            "exact_deduplication": "normalized SHA-256",
        },
        "output": str(docs_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
