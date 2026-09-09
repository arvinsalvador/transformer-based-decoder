# Final implementation audit and remaining execution evidence

This is a source-owned audit explanation, not a measured final experiment report.
Generate `reports/system_audit.json` and `reports/final_summary.json` for the current
workspace. Implementation checks, runtime checks and external GPU verification are
different forms of evidence; absent artifacts are NOT_RUN, never fabricated PASS.

| Area | Implemented safeguard / evidence | Remaining execution proof |
|---|---|---|
| Data flow | Streaming ingestion, canonical preprocessing and hash-linked artifacts | Ingest the selected corpus and retain manifests |
| Reproducibility | Seeds, subset hashes, architecture hash, resolved plan and stage seals | Retain actual final environment and run IDs |
| Leakage | Deduplication before split; train-only fitting; validation selection | Attest no manual test-based tuning; near duplicates are not excluded |
| Document cap | Independent ingestion and total-corpus preflight limits | Verify actual train + validation + test ≤100K |
| WordPiece | Genuine model type, vocabulary and training provenance checks | Fit and verify canonical tokenizer |
| Trigram | Add-k model, disk-backed counts, readonly evaluation | Complete FULL isolated baseline |
| Transformer | Custom causal decoder, strict payload shape and architecture checks | Complete FULL training and best export |
| Training | Causal targets, masking, accumulation, resume, validation-selected weights | Verify actual server precision, memory and checkpoint recovery |
| Evaluation | Common token events, full-test scope, finite metrics, paired saved prompts | Complete shared test evaluation |
| Orchestration | Nested subsets, fixed validation/test, technical gates, FULL primary | Execute explicit dry run, small scale, then FULL |
| Docker | Source-only allowlist, CPU/GPU services, runtime volumes | GPU driver/toolkit and real CUDA test remain external |
| GPU readiness | Central device handling and explicit CUDA requirement | Run server preflight and dry run; CPU tests cannot prove CUDA |
| Artifact separation | Runtime directory exclusions plus weight/SQLite patterns | User controls source review; no Git index inspection was performed |
| UI accuracy | Saved-audit timestamp, separate execution status, no automatic training | Regenerate snapshot after changing artifacts |
| Documentation | Architecture, methodology, GPU runbook and submission/reproducibility checklists | Record actual results and course-specific dataset acceptance |

## Changes found necessary during finalization

- SQLite extensions now have explicit ignore patterns outside runtime directories too.
- Docker includes source audit policy files and documentation, but no corpus or models.
- FULL reporting validates seals, canonical identities, model payload structure, finite
  metrics, summary/matrix consistency and paired generation. Limited evaluations fail.
- Legacy workflow placeholders were removed from implemented navigation.
- The supplied archive contains a 123,708,117-byte CSV. GPU per-file ingestion capacity
  is now 128 MiB; the independent 100,000-document limit and ML methodology are unchanged.
- The runbook extracts that CSV into a dedicated input directory and selects its `text`
  column. It does not mix seed/reference CSVs into the corpus or ingest a ZIP directly.

## Scientific and operational limitations

The raw SUROY README identifies synthetic tourism text. Its advertised document count
is unverified until canonical preprocessing; do not claim scraped social-media data.
Exact deduplication does not establish semantic/template disjointness. One seed,
fixed vocabulary, unequal model contexts and hardware-dependent timing remain explicit.
Missing historical peak RAM is N/A, not final RSS. Hash consistency cannot prove unseen
human decisions. Source scanning is heuristic, not a comprehensive security assessment.
Stage manifests retain absolute runtime roots; cross-root migration is not transparent.
Direct dependencies are declared, but transitive dependencies are not fully locked;
retain the actual environment snapshot. No broad dependency modernization was performed.

Final GPU training was not launched during implementation. No data was deleted and no
Git operations were performed. Generated readiness reports are runtime artifacts and
must not be mistaken for a submission containing measured FULL results.
