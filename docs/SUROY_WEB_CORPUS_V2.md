# SUROY Web Corpus v2

Your v1 result was high quality but small: 435 documents, 6,832 unique word forms, and only 7.39% repeated sentence occurrences.

v2 changes the crawl budget from a shared host counter to a per-seed counter, deepens relevant tourism crawling, filters obvious repeated directory boilerplate, and uses smaller language-model chunks.

Run on the GPU server after Git pull:

```bash
python scripts/collect_suroy_web_corpus.py   --sources resources/suroy_web_sources.csv   --output-dir data/external/suroy_web_v2   --max-total-pages 800   --target-chars 800
```

Then audit `data/external/suroy_web_v2/web_documents.jsonl`.

Do not build the hybrid corpus yet. A useful next target is at least ~2,500 real web documents; that allows a 50/50 hybrid of about 5,000 documents.
