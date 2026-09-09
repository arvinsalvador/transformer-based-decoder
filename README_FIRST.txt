Project SUROY Web Corpus Bundle

Copy the contents of this bundle into your repository root, preserving folders:
  resources/
  scripts/
  docs/

No generated dataset is committed. The GPU server creates runtime files under data/ after git pull.

Suggested commit message:
  feat: add SUROY public web corpus collection and diversity pipeline

Then on GPU server:
  git pull origin dev
  python scripts/collect_suroy_web_corpus.py --sources resources/suroy_web_sources.csv --output-dir data/external/suroy_web --max-total-pages 300
