# Submission and presentation checklist

## Source

- [ ] Include src/, scripts/, app/, config/, tests/, docs/, README, dependency definitions,
  Docker files, Makefile and ignore rules under your own Git control.
- [ ] Run full pytest, Ruff and all CLI help checks; record exact results.
- [ ] Inspect source inventory warnings; review possible secrets without publishing values.

## Runtime artifacts — exclude from ordinary source commits

- [ ] No corpus/split JSONL, tokenizer vocabulary/artifacts, SQLite database, model weights,
  optimizer/checkpoint state, logs, generated matrices/reports or credentials.
- [ ] Preserve .gitkeep directory skeletons; .env remains private.
- [ ] Decide separately how the instructor receives large runtime artifacts and final reports.

## Homework evidence

- [ ] Actual total corpus ≤100K; show train/validation/test counts and topic/provenance.
- [ ] Genuine train-fitted WordPiece and custom causal decoder-only Transformer.
- [ ] Trigram baseline and validated FULL comparison on identical test events.
- [ ] Actual performance, training time, size and hardware; missing metrics labeled N/A.
- [ ] Final report integrity passes; no bounded test subset presented as FULL.
- [ ] Attest no test-based tuning, checkpoint selection or cherry-picked final scale.

## Presentation

- [ ] Requirements, objective, dataset topic/count and pipeline.
- [ ] WordPiece example, trigram formula, Transformer blocks and causal mask.
- [ ] GPU/training setup and shared likelihood methodology.
- [ ] FULL comparison, computational cost and all saved prompt continuations.
- [ ] Supporting scales (including failures), limitations and measured conclusion.
- [ ] Explain fixed tokenizer, one seed, context resets and hardware-dependent timing.
- [ ] Use Final Results presentation view; do not hide integrity or provenance warnings.
