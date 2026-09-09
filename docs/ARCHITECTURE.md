# System architecture

Raw TXT/CSV/PDF/DOCX → streaming ingestion JSONL → conservative normalization,
quality filtering and exact deduplication → deterministic canonical splits.
WordPiece fits only train; validation and test are not vocabulary fitting sources.

The shared vocabulary feeds two distinct learners:

- Trigram: two-token history, add-k probabilities, bounded SQLite counting/querying.
- Transformer: token/learned-position embeddings, pre-norm causal attention and FFN
  residual blocks, final LayerNorm and tied vocabulary head. No pretrained weights.

Training is single-device with token-weighted causal loss, AdamW, warmup/cosine,
accumulation, clipping, validation-selected best weights and resumable checkpoints.
Evaluation streams the same test documents, scores tokens plus EOS once per model,
and saves metrics and identical-prompt generation. It does not train models.

Experiments coordinate these APIs sequentially, isolate scales and retain hash-linked
stage records. FULL is primary; supporting scales are not candidates for choosing a
better-looking final result. Reporting reads and verifies these records without
training, scoring, regenerating examples or selecting a checkpoint by test metrics.

The UI invokes bounded development services or reads summaries. Dashboard reads
saved report metadata only. Final reports are ignored runtime outputs, not source.
