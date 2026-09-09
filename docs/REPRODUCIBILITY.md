# Reproducibility and integrity

Retain the canonical dataset manifest, actual split SHA-256 values, tokenizer source
and artifact hashes, subset seed/selection manifest, architecture fingerprint, project
seed, resolved experiment-plan fingerprint, environment snapshots, training run IDs,
model manifests, evaluation ID and primary experiment ID. These link recorded results
to specific bytes and settings; they do not prove unseen human decisions or guarantee
bit-identical results across GPU architectures/PyTorch versions.

Run the final report on the same runtime roots used for the experiment. Absolute
artifact paths are recorded deliberately; transparent cross-root migration is not
implemented. Do not edit manifests to bypass validation. Use Phase 9 resume for failed
stages; evaluation-only failures should not trigger model retraining.

Atomic replacement is per file. Status/summary are completion authorities, and stage
hashes detect changes. After a hard kill, inspect processes before manually removing
that experiment's stale writer lock. Never remove an active writer's lock.

The final auditor checks FULL status, seals, canonical input provenance, full-test
scope, actual counts, model payload structure, architecture and metric consistency.
Its source audit is static evidence, not a replacement for pytest, CLI checks or a
real GPU dry run. Keep their console outputs with your experiment records.

Report generation writes only beneath REPORT_DIR. Re-running it replaces report
outputs, not datasets/models. Back up previous reports if you need historical versions.
HTML is escaped standalone text; Markdown/JSON/CSV retain the report's actual findings.
