#!/usr/bin/env python3
"""
Build a final hybrid SUROY corpus without padding to an arbitrary document quota.

Policy:
- Keep all unique real/public web documents first.
- Add a deterministic sample of synthetic documents.
- By default, synthetic documents may not exceed 50% of the final corpus.
- Never exceed --max-documents (default 100000).
- If there are not enough real documents, final size is allowed to be below the target.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from pathlib import Path

def norm(text):
    return re.sub(r"\s+", " ", text).strip()

def key(text):
    return hashlib.sha256(norm(text).lower().encode("utf-8")).hexdigest()

def iter_web(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r, dict) and isinstance(r.get("text"), str):
                yield r

def iter_synthetic(path):
    path = Path(path)

    if path.is_dir():
        files = sorted(path.glob("*.csv"))

        if not files:
            raise ValueError(
                f"No CSV files found in synthetic corpus directory: {path}"
            )

        print(f"Synthetic corpus parts found: {len(files)}")

        for csv_path in files:
            print(f"Reading synthetic part: {csv_path}")

            with csv_path.open(
                encoding="utf-8",
                newline=""
            ) as f:
                for row in csv.DictReader(f):
                    if row.get("text"):
                        yield row

    else:
        with path.open(
            encoding="utf-8",
            newline=""
        ) as f:
            for row in csv.DictReader(f):
                if row.get("text"):
                    yield row

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--web", default="data/external/suroy_web/web_documents.jsonl")
    p.add_argument("--synthetic", required=True, help="Original synthetic 100K CSV")
    p.add_argument("--output", default="data/raw/suroy_final/suroy_hybrid.csv")
    p.add_argument("--max-documents", type=int, default=100000)
    p.add_argument("--max-synthetic-fraction", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not (1 <= args.max_documents <= 100000):
        raise SystemExit("--max-documents must be 1..100000")
    if not (0 <= args.max_synthetic_fraction < 1):
        raise SystemExit("--max-synthetic-fraction must be >=0 and <1")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise SystemExit("Output exists; use --overwrite only intentionally")

    seen = set()
    real = []
    for r in iter_web(args.web):
        text = norm(r["text"])
        if len(text) < args.min_chars:
            continue
        h = key(text)
        if h in seen:
            continue
        seen.add(h)
        real.append({
            "text": text,
            "provenance_type": "real_public_web",
            "source_url": r.get("source_url",""),
            "source_domain": r.get("source_domain",""),
            "source_title": r.get("source_title",""),
            "region": r.get("region",""),
            "category": r.get("category",""),
            "synthetic": "false",
        })

    real = real[:args.max_documents]

    # Mathematical cap: synthetic <= frac * final.
    # With R real docs and fraction f, max synthetic is floor(R*f/(1-f)).
    if args.max_synthetic_fraction == 0 or not real:
        synthetic_cap = 0
    else:
        synthetic_cap = int(len(real) * args.max_synthetic_fraction / (1 - args.max_synthetic_fraction))
    synthetic_cap = min(synthetic_cap, args.max_documents - len(real))

    rng = random.Random(args.seed)
    reservoir = []
    n_seen = 0
    for r in iter_synthetic(args.synthetic):
        text = norm(r["text"])
        if len(text) < args.min_chars or key(text) in seen:
            continue
        n_seen += 1
        item = {
            "text": text,
            "provenance_type": r.get("provenance_type","synthetic_augmented"),
            "source_url": r.get("source_url",""),
            "source_domain": "",
            "source_title": r.get("source_title",""),
            "region": r.get("region_focus", r.get("province","")),
            "category": r.get("category",""),
            "synthetic": "true",
        }
        if len(reservoir) < synthetic_cap:
            reservoir.append(item)
        elif synthetic_cap:
            j = rng.randrange(n_seen)
            if j < synthetic_cap:
                reservoir[j] = item

    rows = real + reservoir
    rng.shuffle(rows)
    fields = ["document_id","text","provenance_type","source_url","source_domain","source_title","region","category","synthetic"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"document_id": f"SUROY-FINAL-{i:06d}", **r})

    summary = {
        "output": str(out),
        "documents": len(rows),
        "real_documents": len(real),
        "synthetic_documents": len(reservoir),
        "synthetic_fraction": round(len(reservoir) / max(1, len(rows)), 4),
        "max_documents": args.max_documents,
        "note": "Final size is not padded; quality takes priority over reaching 100000.",
    }
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
