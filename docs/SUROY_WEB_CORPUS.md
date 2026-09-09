# Project SUROY Web + Hybrid Corpus Pipeline

## Why this exists

The first 100K SUROY corpus successfully tested the ingestion/tokenizer pipeline, but it was too templated for a convincing final language-model experiment (only ~635 unique whitespace-delimited word forms in ~80K training documents).

This bundle adds a **real/public web corpus collection stage** and a **quality-first hybrid builder**.

## Important rule

Do **not** automate Facebook scraping. The collector intentionally skips Facebook/Instagram/TikTok. Public business/tourism Facebook posts can be added later only through permissioned exports or manual curation as a separate provenance type.

## Repo files to add

Copy these into the same paths in your repository:

- `resources/suroy_web_sources.csv`
- `scripts/collect_suroy_web_corpus.py`
- `scripts/build_suroy_hybrid_corpus.py`
- `scripts/audit_corpus_diversity.py`
- `docs/SUROY_WEB_CORPUS.md`

The generated runtime corpus remains under `data/` and should stay ignored by Git.

## GPU server workflow after Git pull

### 1. Collect public web text

```bash
python scripts/collect_suroy_web_corpus.py   --sources resources/suroy_web_sources.csv   --output-dir data/external/suroy_web   --max-total-pages 300
```

Inspect:

```bash
cat data/external/suroy_web/collection_manifest.json
wc -l data/external/suroy_web/web_documents.jsonl
```

A page count is not a document count: each useful page can produce several paragraph-sized language-model documents.

### 2. Audit real web corpus diversity

```bash
python scripts/audit_corpus_diversity.py   data/external/suroy_web/web_documents.jsonl
```

### 3. Build the hybrid corpus

Point `--synthetic` to the original synthetic 100K CSV stored outside Git:

```bash
python scripts/build_suroy_hybrid_corpus.py   --web data/external/suroy_web/web_documents.jsonl   --synthetic "$HOME/datasets/suroy_100k/suroy_tourism_corpus_100k.csv"   --output data/raw/suroy_final/suroy_hybrid.csv   --max-documents 100000   --max-synthetic-fraction 0.50
```

The script intentionally **does not pad** to 100K. If only 4,000 strong real web documents are collected, a 50% synthetic cap yields at most about 8,000 total documents. That is preferable to inflating the corpus with repeated templates.

### 4. Audit final corpus

```bash
python scripts/audit_corpus_diversity.py   data/raw/suroy_final/suroy_hybrid.csv
```

Suggested quality checks before final WordPiece training:

- unique word forms: preferably several thousand, not hundreds;
- repeated sentence occurrence ratio: preferably below 0.20;
- no accidental 100K quota padding;
- real/public provenance retained for every real web chunk.

### 5. Start a fresh canonical ingestion

Do not mix this with the old Phase 2/3 artifacts.

Use new output paths first:

```bash
python scripts/ingest_dataset.py   --source data/raw/suroy_final   --output data/processed/suroy_final_documents.jsonl   --config config/gpu.yaml   --csv-mode rows   --text-columns text
```

Then prepare canonical splits. Because the existing `data/splits/*` belong to the old corpus, explicitly replace them only after you have archived or intentionally discarded the old test corpus:

```bash
mkdir -p data/archive_synthetic_pipeline_test
cp -a data/splits data/archive_synthetic_pipeline_test/splits 2>/dev/null || true
cp -a models/tokenizer data/archive_synthetic_pipeline_test/tokenizer 2>/dev/null || true
```

Then rebuild:

```bash
python scripts/prepare_dataset.py   --input data/processed/suroy_final_documents.jsonl   --clean-output data/processed/suroy_final_clean.jsonl   --split-dir data/splits   --config config/gpu.yaml   --overwrite
```

Train a **new tokenizer** with `--overwrite`; never reuse the old 1,726-token tokenizer for the new corpus.

## Academic wording

Recommended:

> The initial synthetic corpus was retained as a pipeline-validation dataset. Diversity auditing showed excessive lexical/template repetition, so the final corpus was rebuilt using naturally written public tourism information from curated government and travel sources. A bounded amount of synthetic text was then added with explicit provenance. The final corpus size was allowed to remain below the 100,000-document maximum rather than padding the corpus with near-duplicate synthetic records.

## Source selection

The source catalog prioritizes:
- Department of Tourism destination pages;
- Province of Surigao del Norte municipality/tourism pages;
- Surigao City tourism pages;
- Surigao del Sur provincial tourism pages;
- naturally written Siargao-focused local travel guides.

The collector preserves the source URL for every document so the corpus remains auditable.
