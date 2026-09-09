# GPU execution runbook — documentation, not an automatic job

Run from the repository root on the GPU server. Obtain source under your own Git
control. Install a compatible NVIDIA host driver, Docker and NVIDIA Container Toolkit;
never install the host driver inside the container. The GPU service uses CUDA 12.8
PyTorch wheels; consult the existing README compatibility guidance for the host driver.
Use a user-managed tmux/screen session if desired. No command below was run on real
homework data during implementation. Replace the two indicated IDs with actual outputs.

## 1. Verify host/container and prepare volumes

```bash
nvidia-smi
docker version
mkdir -p data/raw data/processed data/splits models checkpoints experiments reports
docker compose -f docker-compose.gpu.yml config
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm app python -c \
  "import torch; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Place your chosen raw corpus in data/raw. Verify permissions for the configured
container UID/GID. Ingest no more than 100,000 total documents; filtering can reduce
the accepted corpus. Set CSV columns explicitly if required by your dataset.
ZIP archives are not ingested directly: extract the intended corpus on the server
first into a dedicated directory containing only that corpus CSV. The ingestion CLI
requires a directory; do not give it a filename or a mixed seed/reference directory.
The current raw SUROY package describes synthetic tourism documents, not scraped
Facebook posts. Disclose that provenance and confirm it suits your course requirements.
Its advertised 100K count is not a verified clean-corpus count.
The archive lists a 123,708,117-byte CSV (about 118 MiB). The GPU profile permits
128 MiB per source file; the independent 100,000-document limit remains unchanged.

## 2. Prepare canonical data and WordPiece

For the currently supplied archive, extract the single corpus CSV without overwriting
an existing file (requires the host's unzip utility):

```bash
unzip -n data/raw/suroy_tourism_corpus_100k_only.zip -d data/raw/corpus
```

```bash
docker compose -f docker-compose.gpu.yml run --rm app python scripts/ingest_dataset.py \
  --source data/raw/corpus \
  --csv-mode rows --text-columns text \
  --output data/processed/raw_documents.jsonl --config config/gpu.yaml --limit 100000
docker compose -f docker-compose.gpu.yml run --rm app python scripts/prepare_dataset.py \
  --input data/processed/raw_documents.jsonl --config config/gpu.yaml --limit 100000
```

Record the completed preprocessing manifest path printed by the command:

```bash
DATASET_MANIFEST=experiments/preprocessing/REPLACE_WITH_ACTUAL_RUN.json
docker compose -f docker-compose.gpu.yml run --rm app python scripts/train_tokenizer.py \
  --train data/splits/train.jsonl --dataset-manifest "$DATASET_MANIFEST" --config config/gpu.yaml
docker compose -f docker-compose.gpu.yml run --rm app python scripts/inspect_transformer.py \
  --tokenizer models/tokenizer --config config/gpu.yaml
```

Do not fit vocabulary on validation/test. No test analysis is needed before training.
Phase 9 trains the isolated trigram automatically; a separate global baseline is
optional, not a prerequisite. If specifically wanted, its existing command is:

```bash
docker compose -f docker-compose.gpu.yml run --rm app python scripts/train_trigram.py \
  --train data/splits/train.jsonl --tokenizer models/tokenizer --config config/gpu.yaml
```

## 3. Plan, preflight, dry run, then selected training

```bash
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --plan
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --preflight
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --dry-run-only
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --execute --scale 1000
```

Inspect the returned experiment ID, statuses, metrics, memory and checkpoints. Confirm
that your final training configuration has no unintended development step limit.
Stop and start a new plan if architecture/batch/LR changes are required. FULL is a
deliberate later command, not automatically chained to the smoke test:

```bash
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --execute --full-only --confirm-large-run
```

Append `--resume ACTUAL_EXPERIMENT_ID` to continue the SAME experiment, including
adding FULL to an earlier scale's run. Do not resume another scale's checkpoint.

## 4. Final evaluation and report

FULL orchestration already performs the shared full-test evaluation. If that stage
failed after models completed, the following resumes evaluation without retraining:

```bash
EXPERIMENT_ID=REPLACE_WITH_ACTUAL_EXPERIMENT_ID
docker compose -f docker-compose.gpu.yml run --rm app python scripts/run_experiments.py \
  --config config/gpu.yaml --experiment-config config/experiments.yaml \
  --dataset-manifest "$DATASET_MANIFEST" --execute --full-only --confirm-large-run \
  --resume "$EXPERIMENT_ID"
docker compose -f docker-compose.gpu.yml run --rm app python scripts/generate_final_report.py \
  --config config/gpu.yaml --experiment "$EXPERIMENT_ID" --output reports
```

Do not point standalone evaluation at global models and assume they are the isolated
FULL artifacts. The orchestrator supplies the subset provenance and isolated paths.
Use its evaluation-only recovery to preserve the recorded comparison and stage seals.

Before any final experiment, generate readiness documentation without training:

```bash
docker compose -f docker-compose.gpu.yml run --rm app python scripts/generate_final_report.py \
  --config config/gpu.yaml --readiness-only --output reports
```

For direct Linux Python execution, use the same script arguments after installing the
declared GPU dependencies. GPU profile explicitly requires CUDA for training; reporting
can explain its absence without trying to train. Review audit failures and CPU/bounded
run warnings before presentation. Back up runtime artifacts; do not commit them by default.
