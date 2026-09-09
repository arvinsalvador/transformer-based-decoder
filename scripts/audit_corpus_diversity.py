#!/usr/bin/env python3
"""Measure lexical and structural diversity for CSV or JSONL corpora."""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?")
SENT_RE = re.compile(r"(?<=[.!?])\s+")

def records(path, text_column):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get(text_column):
                    yield row[text_column]
    else:
        with p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if isinstance(r, dict) and isinstance(r.get(text_column), str):
                    yield r[text_column]

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--text-column", default="text")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    words = Counter()
    sentences = Counter()
    docs = 0
    total_words = 0
    lengths = []
    for text in records(args.path, args.text_column):
        if args.limit and docs >= args.limit:
            break
        docs += 1
        toks = [x.lower() for x in WORD_RE.findall(text)]
        words.update(toks)
        total_words += len(toks)
        lengths.append(len(toks))
        for s in SENT_RE.split(re.sub(r"\s+", " ", text.strip())):
            s = re.sub(r"\s+", " ", s).strip().lower()
            if len(s) >= 40:
                sentences[s] += 1

    repeated_occ = sum(n for n in sentences.values() if n > 1)
    repeated_ratio = repeated_occ / max(1, sum(sentences.values()))
    hapax = sum(1 for n in words.values() if n == 1)
    summary = {
        "documents": docs,
        "total_word_tokens": total_words,
        "unique_word_forms": len(words),
        "type_token_ratio": round(len(words) / max(1, total_words), 6),
        "hapax_word_forms": hapax,
        "avg_words_per_document": round(total_words / max(1, docs), 2),
        "unique_sentences": len(sentences),
        "repeated_sentence_occurrence_ratio": round(repeated_ratio, 4),
        "quality_flags": {
            "unique_words_below_2000": len(words) < 2000,
            "high_sentence_repetition": repeated_ratio > 0.20,
        },
        "top_30_words": words.most_common(30),
        "top_10_repeated_sentences": sentences.most_common(10),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
