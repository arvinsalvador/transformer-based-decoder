# SUROY Web Corpus v3

v2 produced 665 documents with good lexical diversity, but some real-estate/directory noise appeared (`property`, `lot`, `services` among frequent terms).

v3:
- adds more independent Siargao/Surigao travel and official sources;
- filters real-estate/classified language;
- requires each emitted chunk to contain both a Surigao/Siargao location term and a tourism/activity term;
- keeps per-seed page limits and public-source provenance;
- adds a merger for v1 + v2 + v3 with exact and conservative near-duplicate removal.

## GPU commands after Git pull

Collect v3 into a new directory:

```bash
python scripts/collect_suroy_web_corpus_v3.py   --sources resources/suroy_web_sources_v3.csv   --output-dir data/external/suroy_web_v3   --max-total-pages 1200   --target-chars 700
```

Audit v3:

```bash
python scripts/audit_corpus_diversity.py data/external/suroy_web_v3/web_documents.jsonl
```

Merge all real corpora:

```bash
python scripts/merge_suroy_web_corpora.py   data/external/suroy_web/web_documents.jsonl   data/external/suroy_web_v2/web_documents.jsonl   data/external/suroy_web_v3/web_documents.jsonl   --output data/external/suroy_web_merged/web_documents.jsonl
```

Audit merged:

```bash
python scripts/audit_corpus_diversity.py data/external/suroy_web_merged/web_documents.jsonl
```

Do not build the synthetic hybrid until the merged real-corpus count/diversity is reviewed.
