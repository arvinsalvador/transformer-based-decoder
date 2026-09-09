# SUROY Real-Web Cleanup

Run after merging v1+v2+v3:

```bash
python scripts/clean_suroy_real_corpus.py \
  --input data/external/suroy_web_merged/web_documents.jsonl \
  --output data/external/suroy_web_clean/web_documents.jsonl \
  --max-sentence-occurrences 5
```

Then audit:

```bash
python scripts/audit_corpus_diversity.py \
  data/external/suroy_web_clean/web_documents.jsonl
```

Accept only if repeated_sentence_occurrence_ratio is below 0.20, preferably below ~0.12.
Keep several thousand unique word forms. Prefer >3,125 clean real documents so a 50/50
hybrid can exceed ~6,250 total docs, leaving about 5,000 train docs after an 80/10/10 split.
