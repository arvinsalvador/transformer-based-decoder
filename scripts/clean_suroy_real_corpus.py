#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’.-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?")

KNOWN_NOISE = (
    "elevating siargao business discovery with refined simplicity",
    "your trusted local business directory for siargao island areas",
    "send it to socials, chats, or copy the link",
    "learn how your comment data is processed",
    "affiliate note:",
    "direct booking offer:",
    "local booking platform built in the philippines",
    "we use cookies to improve your experience",
    "property for sale", "lot for sale", "house and lot", "real estate",
    "marketing and sales", "classified", "property for rent",
    "sqm beach front", "sq.m beach front", "for rent gl siargao",
)

REGION = (
    "siargao","surigao","general luna","dapa","del carmen","pilar","san isidro",
    "santa monica","burgos","san benito","socorro","bucas grande","sohoton",
    "sugba","cloud 9","magpupungko","pacifico","hinatuan","britania","cagwait","bislig"
)
TOURISM = (
    "tour","travel","tourism","destination","attraction","beach","surf","island",
    "lagoon","cave","falls","river","mangrove","itinerary","hotel","resort",
    "restaurant","transport","ferry","airport","snorkel","diving","kayak",
    "visitor","guide","heritage","museum"
)

def norm(s): return re.sub(r"\s+", " ", s).strip()
def nsent(s):
    s = norm(s).lower()
    s = re.sub(r"[“”\"'’`]+", "", s)
    return norm(s)
def dhash(s): return hashlib.sha256(norm(s).lower().encode()).hexdigest()
def sentences(text): return [norm(x) for x in SENT_SPLIT.split(text) if norm(x)]

def nav_like(s):
    low = s.lower()
    words = WORD_RE.findall(low)
    if len(words) < 4:
        return True
    if s.count("/") >= 3 and len(words) < 45:
        return True
    punct = sum(s.count(x) for x in ",;:|·")
    verbs = (" is "," are "," can "," will "," has "," have "," offers "," provides ",
             " takes "," includes "," visit "," travel "," explore "," enjoy "," located ")
    return punct >= 5 and len(words) < 35 and not any(v in f" {low} " for v in verbs)

def topic_ok(s):
    low = s.lower()
    return any(x in low for x in REGION) and any(x in low for x in TOURISM)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/external/suroy_web_clean/web_documents.jsonl")
    ap.add_argument("--max-sentence-occurrences", type=int, default=5)
    ap.add_argument("--min-sentence-chars", type=int, default=35)
    ap.add_argument("--min-document-chars", type=int, default=180)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    inp, out = Path(a.input), Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.overwrite:
        raise SystemExit("Output exists; use --overwrite intentionally")

    records = []
    with inp.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r, dict) and isinstance(r.get("text"), str):
                records.append(r)

    freq = Counter()
    for r in records:
        for s in sentences(r["text"]):
            n = nsent(s)
            if len(n) >= a.min_sentence_chars:
                freq[n] += 1

    repeated = {s for s,c in freq.items() if c > a.max_sentence_occurrences}
    seen = set()
    written = 0
    stats = Counter()

    with out.open("w", encoding="utf-8") as sink:
        for r in records:
            original = norm(r["text"])
            kept = []
            removed = 0
            for s in sentences(original):
                low = s.lower()
                n = nsent(s)
                if any(x in low for x in KNOWN_NOISE):
                    stats["removed_known_noise"] += 1; removed += 1; continue
                if len(n) >= a.min_sentence_chars and n in repeated:
                    stats["removed_repeated"] += 1; removed += 1; continue
                if nav_like(s):
                    stats["removed_navigation"] += 1; removed += 1; continue
                kept.append(s)

            cleaned = norm(" ".join(kept))
            if not cleaned:
                stats["dropped_empty"] += 1; continue
            if len(cleaned) < a.min_document_chars:
                stats["dropped_short"] += 1; continue
            if not topic_ok(cleaned):
                stats["dropped_off_topic"] += 1; continue

            h = dhash(cleaned)
            if h in seen:
                stats["exact_duplicates_after_cleaning"] += 1; continue
            seen.add(h)

            o = dict(r)
            written += 1
            o["document_id"] = f"WEB-CLEAN-{written:07d}"
            o["text"] = cleaned
            o["clean_removed_sentences"] = removed
            o["clean_original_chars"] = len(original)
            o["clean_final_chars"] = len(cleaned)
            sink.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_records": len(records),
        "output_records": written,
        "unique_sentence_forms": len(freq),
        "sentences_above_frequency_threshold": len(repeated),
        "max_sentence_occurrences": a.max_sentence_occurrences,
        **stats,
        "output": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()
