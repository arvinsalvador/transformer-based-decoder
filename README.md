# Transformer-Based Decoder-Only Language Model

University machine-learning homework. **Status: Phase 8 — Shared evaluation and comparison.**
Ingestion, preprocessing, WordPiece, trigram scoring/generation, the custom Transformer,
single-device training, and shared evaluation are implemented. Final experiments remain pending.

## Homework requirements

- Train a decoder-only Transformer using a WordPiece tokenizer.
- Use a chosen topic with **at most 100,000 documents**. This is a maximum, not a minimum.
- Compare model performance and training time against a trigram language model.
- Start development with small subsets; use the GPU profile for final experiments later.

Document count alone does not describe dataset size. Tokenizer analysis and training report
token counts. The local profile allows at most 1,000 candidate records per ingestion run.
No corpus is loaded until ingestion is explicitly requested.
The random seed is applied to splitting and actual training; no simulated metrics are used.

## Architecture and directories

The UI calls centralized configuration/device utilities and the same ingestion API
as the CLI. Ingestion has no Streamlit or PyTorch dependency at import time.
Evaluation shares canonical components without calling any training API.

```text
app/                  Streamlit entry point, workflow pages and bounded development controls
src/config/           YAML loading, validation, environment and path resolution
src/utils/            Structured CPU/CUDA diagnostics
src/data/             Ingestion plus preprocessing, deduplication, and canonical split modules
src/tokenizer/        Phase 4 WordPiece corpus, training, statistics, and load API
src/trigram/          WordPiece trigram counting, SQLite persistence, scoring and generation
src/transformer/      Custom causal decoder model, factory and architecture inspection
src/training/         Streaming causal sequences, training, precision, checkpoints and monitoring
src/evaluation/       Shared read-only scoring, generation, comparison and artifact exports
config/               local.yaml and gpu.yaml
scripts/              Nine CLIs covering ingestion through shared model evaluation
tests/                Phase 1–8 unit, integration, CLI and Streamlit regression tests
data/raw/             Original TXT/CSV/DOCX/PDF sources; uploads stored in unique subdirectories
data/processed/       Raw extracted JSONL; NOT cleaned text
data/splits/          Canonical Phase 3 train/validation/test JSONL
models/tokenizer/     Generated tokenizer files
models/trigram/       Generated baseline artifacts
models/transformer/   Generated weights
checkpoints/          Resumable per-run training state
experiments/          Run metadata, incremental histories and summaries
reports/              Generated reports
```

Runtime directories contain only `.gitkeep` placeholders initially. Store datasets,
uploads, tokenizer outputs, weights, checkpoints, logs, and reports in these ignored
locations. Large generated artifacts are excluded from version control because they
are expensive to transfer, may contain private documents, and are reproduced from
code/configuration. Retain data provenance and experiment recipes as reviewed source
configuration in later phases. Never commit `.env`, credentials, caches or virtual
environments. Uploaded filenames are untrusted: uploads use generated identifiers and
sanitized basenames. Extracted text is rendered as plain text, never executable HTML.
The UI executes no shell inputs and introduces no pickle loading.

## Local setup (WSL recommended)

Use Python **3.12** (supported range 3.11–3.13), an isolated environment, and optionally
Docker Desktop with its WSL 2 backend and Ubuntu integration enabled. No NVIDIA GPU
or CUDA installation is required. An approximately 8 GB RAM machine should use the
CPU wheel and small subsets. Docker dependency installation needs disk space and
network access; it does not download models or datasets.

Run commands from the repository root inside WSL. Install Python 3.12 and its venv
support through your Ubuntu distribution if available; otherwise use a Python version
manager or the Docker workflow below. Do not reuse a Windows virtual environment in WSL.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>. Stop the server with Ctrl+C. The recommended module
invocation from the root ensures both `app` and `src` are importable. Native Windows
works too: use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`, followed by
the same `python -m ...` commands. Make is optional.

`requirements.txt` pins direct runtime dependencies. CPU/CUDA Torch builds are selected
by installing Torch from the appropriate official index **before** the shared requirements.
Transitive dependencies and Docker base images are not fully locked; capture a full
environment manifest for final controlled experiments. PyMuPDF and python-docx now support
PDF/DOCX extraction. CSV uses Python's streaming csv module. The pinned Hugging Face
`tokenizers` package provides local WordPiece training. No OCR software, LibreOffice,
GUI, database or pretrained model is required.

## Configuration

Precedence: explicit `load_settings(path)` argument → `CONFIG_PATH` → `APP_ENV`
(`local` by default). Process environment overrides optional root `.env` values;
the loader does not modify the global environment. Copy `.env.example` to `.env`
yourself only if needed. Leaving `CONFIG_PATH` unset allows `APP_ENV` to select a profile.

| Setting | Local | GPU |
| --- | --- | --- |
| Device | auto | cuda |
| Maximum / working documents | 100000 / 1000 | 100000 / 100000 |
| Vocabulary | 8000 | 12000 |
| Context / embedding | 128 / 192 | 256 / 256 |
| Layers / heads / feedforward | 3 / 4 / 768 | 4 / 4 / 1024 |
| Dropout | 0.1 | 0.1 |
| Batch / epochs / accumulation | 4 / 2 / 1 | 16 / 3 / 1 |
| Mixed precision / workers | false / 1 | true / 4 |
| Random seed | 42 | 42 |

These are initial settings, not validated training recommendations for any GPU.
Edit YAML for model/training parameters. `DEVICE`, `WORKING_DOCUMENT_LIMIT`, and
`BATCH_SIZE` provide selected environment overrides and undergo the same validation.
Unknown or missing YAML fields are rejected to catch typos. Positive integer fields,
the 100,000-document cap, working limit versus maximum, dropout, booleans, seed range,
and embedding/head divisibility are checked before later phases can allocate resources.

`DATA_DIR`, `MODEL_DIR`, `CHECKPOINT_DIR`, `EXPERIMENT_DIR`, and `REPORT_DIR` override
runtime paths. Relative paths resolve against the project root regardless of the current
directory; absolute external storage paths are also accepted. The loader never creates
directories or reads their contents. Keep this repository checkout for configuration
discovery; standalone wheel distribution is outside Phase 1.

`auto` uses CUDA when available, otherwise CPU. `cpu` forces CPU. `cuda` raises a
controlled error if unavailable. The UI displays the error and CPU diagnostics without
claiming the CUDA request succeeded. Mixed precision potential means a selected CUDA
device with compute capability >= 7; it is not a benchmark or guarantee, and does not
enable mixed precision. CPU mixed precision is outside this initial policy.

## Docker CPU development

```bash
docker compose config --quiet
docker compose up --build -d
docker compose logs app
docker compose ps
docker compose down
```

This builds Python 3.12 slim with CPU-only Torch, explicitly selects the local CPU
environment, mounts code/configuration for development, and persists runtime directories.
Port 8501 binds only to localhost. The container runs as UID/GID 1000 by default;
on Linux/WSL use `export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` if your identity differs.
Bind-mounted directories must be writable by that user. Docker `.env` interpolation
controls these IDs and `STREAMLIT_PORT`; application profile/device are explicit in each Compose file.
To override other app settings in Docker, add them to the service environment.

The build context uses an allowlist and excludes all datasets, secrets, generated model
files and local environments. CPU and GPU builds reuse one Dockerfile but select different
Torch wheel indexes. The GPU wheel supplies CUDA runtime libraries; a CUDA compiler/devel
image is unnecessary for this phase. The test stage is excluded from the final runtime.

## GPU server prerequisites and usage

Use a compatible NVIDIA GPU, a Linux host with a functioning NVIDIA driver, Docker
Engine plus Compose, and the **NVIDIA Container Toolkit** configured for Docker.
Validate host `nvidia-smi` before trying the container. This project pins Torch 2.10.0
with the official CUDA 12.8 (`cu128`) wheel. Prefer a current driver; CUDA 12.8 GA's
toolkit driver baseline is Linux 570.26 or newer. NVIDIA compatibility modes may allow
older drivers, but are not assumed here. Confirm the selected wheel supports your GPU
architecture before final experiments. Installing a CUDA wheel alone cannot supply a
host NVIDIA driver or grant container GPU access.

Authoritative references: [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/),
[Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).

After installing/configuring these prerequisites using the vendor instructions:

```bash
nvidia-smi
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm app python -c 'from src.utils.device import detect_device; print(detect_device("cuda"))'
docker compose -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.gpu.yml logs app
docker compose -f docker-compose.gpu.yml down
```

The GPU file is **standalone**, not an override to combine with the CPU file. It reserves
one GPU, sets the GPU profile, and mounts data, models, checkpoints, experiments and
reports persistently. It launches Streamlit; the Phase 7 training CLI can also run in this
service. Use an SSH tunnel for a remote dashboard:
`ssh -L 8501:localhost:8501 user@gpu-server`, then open localhost:8501 locally.

Without Docker, create the same Python environment on the GPU server, install
`torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128` using `python -m pip install`,
install `requirements.txt`, then run:

```bash
APP_ENV=gpu python -m streamlit run app/streamlit_app.py
```

Ensure `CONFIG_PATH` is unset or points to `config/gpu.yaml`.

## Testing and convenience commands

```bash
python -m pytest
python -m pytest --cov=src --cov=app --cov-report=term-missing
python -m ruff check .
python -c 'from src.config.settings import load_settings; from src.utils.device import detect_device; print(load_settings().values); print(detect_device())'
```

Tests mock CUDA metadata and require no GPU. Streamlit AppTest executes the dashboard,
each future section, configuration error handling, and the missing-CUDA warning.
Container-based checks if a supported local Python is unavailable:

```bash
docker build --target test -t homework3-phase4-test .
docker run --rm homework3-phase4-test
docker run --rm homework3-phase4-test python -m ruff check .
```

Optional Make aliases: `make install` (CPU runtime), `make install-dev`, `make test`,
`make lint`, `make run`, `make docker-up`, `make docker-down`, `make gpu-up`, and
`make gpu-down`. Their underlying commands are shown above and in the Makefile.
On an already configured GPU virtual environment, install shared requirements directly;
the Make install targets are intended for local CPU setup.

## Troubleshooting

- No matching wheel: use Python 3.12 and the documented index; Python 3.14 is outside
  this project's supported range.
- Docker unavailable in WSL: start Docker Desktop and enable the distribution's WSL integration.
- CUDA unavailable: inspect `nvidia-smi`, Toolkit setup, wheel variant, and GPU reservation.
  Use the local profile on this CPU-only development machine.
- Configuration error: inspect the named field, YAML types, and environment overrides.
- Import failure: activate the right environment and launch with `python -m streamlit`
  from the repository root.
- Port occupied: choose another host port, e.g. `STREAMLIT_PORT=8503 docker compose up --build -d`,
  then open <http://localhost:8503>. The container still listens internally on port 8501.
- Permission denied for artifacts: match `LOCAL_UID`/`LOCAL_GID` to the directory owner.
- Memory pressure: stop unused containers, keep CPU development subsets small, and
  reduce future batch/context settings. Ingestion processes one source/row at a time and creates no tensors.
- Dependency download failure: verify access to PyPI and download.pytorch.org, then
  retry the build. Do not substitute a CUDA build on a low-memory local machine.

## Remaining phases

3. Cleaning, preparation and dataset splits.
4. WordPiece tokenizer.
5. Trigram baseline.
6. Decoder-only Transformer.
7. Training, resource controls, mixed precision and checkpoints.
8. Evaluation framework.
9. Controlled final experiments and comparisons.
10. Final UI, reporting and validation.

Phase 4 stops at tokenization. No language-model training results or fabricated metrics are present.

## Phase 5 trigram baseline

Phase 5 adds a CPU-only, reproducible baseline over the *same* Phase 3 splits and Phase 4
WordPiece IDs that the future Transformer will use. It estimates
`P(t_i | t_(i-2), t_(i-1))`. Each document is independently modeled as
`[BOS] [BOS] tokens [EOS]`; `[PAD]` is never added and no trigram crosses a document boundary.

Counts are unigram, bigram, and trigram counts. Add-k (Lidstone) smoothing prevents a
zero probability: `(count(w1,w2,w3)+k)/(count(w1,w2)+k*|V|)`, where `|V|` is the actual
serialized tokenizer vocabulary size. Scoring is read-only and reports log likelihood,
average negative log likelihood, and `exp(NLL)` perplexity, including EOS prediction events.
Training time starts at the train-split count pass and ends when the artifact is ready; it
does not include ingestion, preprocessing, or tokenizer fitting. Generation uses greedy or
seeded sampling, short-prompt BOS contexts, EOS/max-token stopping, and canonical tokenizer decoding.

The local profile uses memory counting; the GPU/server profile selects SQLite. Both persist a
safe SQLite count artifact (`models/trigram/trigram_counts.sqlite`) with parameterized SQL and
transactions—never pickle. `trigram_manifest.json` records tokenizer/dataset/split fingerprints,
smoothing, counts, timing, and artifact size. Generated models remain ignored by Git.

SQLite counting now updates disk tables during the document pass. Its write buffers hold at
most 4,096 count updates in total and flush in one transaction at that threshold or at the end
of a document. The SQLite cache is limited to approximately 4 MiB; tokenization still retains
one document. Primary keys support context-prefix queries without a redundant index. Unique
counts and event totals use SQL aggregates. Loading either persisted backend returns a read-only
SQLite model: scoring uses indexed lookups, and generation fetches one context's continuations
(or the vocabulary-bounded unigram fallback). Use `with load_model(path) as model:` to close it.
`auto` deterministically selects memory for the local profile and SQLite for the GPU profile;
manifests record this actual choice. Equal-probability generation ties use ascending token IDs.

Training builds the DB, statistics, and manifest in a unique staging directory. It closes and
checks the DB and completes optional validation before publishing. Without `--overwrite`, an
existing artifact is rejected. With it, same-filesystem renames replace the files; caught failures
restore the previous artifact set and clean staging files. Use exclusive writer access and close
readers before replacement. Individual renames are atomic, but the three-file publication is not
a crash-proof transaction: a process kill or power loss during publication can leave staging
backups requiring recovery. Training duration excludes optional validation and publication.

```bash
python scripts/train_trigram.py --train data/splits/train.jsonl --validation data/splits/validation.jsonl --tokenizer models/tokenizer --output models/trigram --config config/local.yaml
python scripts/evaluate_trigram.py --model models/trigram --test data/splits/test.jsonl --tokenizer models/tokenizer --config config/local.yaml
python scripts/generate_trigram.py --model models/trigram --tokenizer models/tokenizer --prompt "Artificial intelligence" --max-new-tokens 50 --strategy greedy --config config/local.yaml
```

For final server experiments use the identical commands with `--config config/gpu.yaml`; this
does not use CUDA. The Trigram Model Streamlit page is artifact-backed and warns when no model
has been trained. It does not retain a corpus in session state.

Limitations are intentional: the model sees only two previous tokens, has sparse count tables,
cannot represent long-range semantics/coherence, and must fall back for unseen contexts. These are
the reasons it is a useful baseline rather than a claim that a future Transformer is already better.

## Phase 6 decoder-only Transformer architecture

Phase 6 provides a scratch-initialized PyTorch decoder-only architecture, not a training
engine. Its WordPiece vocabulary size and `[PAD]` ID come from the canonical tokenizer artifact,
never the requested YAML vocabulary size. The forward API accepts `LongTensor [B, T]` and returns
unnormalized logits `[B, T, V]`; it does not apply softmax. Phase 7 constructs shifted
next-token targets and uses `ignore_index=-100` for padded targets.

```text
WordPiece IDs → token embeddings + learned positions → dropout
    → [LayerNorm → causal multi-head attention → residual
       LayerNorm → GELU feed-forward → residual] × N
    → final LayerNorm → tied-or-untied vocabulary projection → logits
```

The blocks use pre-norm residual structure. Attention explicitly projects Q/K/V, splits heads,
uses PyTorch scaled-dot-product attention, merges heads, and projects back to the embedding
dimension. Its causal mask allows only current/past positions:

```text
      1 2 3 4
1     ✓ × × ×
2     ✓ ✓ × ×
3     ✓ ✓ ✓ ×
4     ✓ ✓ ✓ ✓
```

An optional `attention_mask` additionally prevents attending to padded key positions; it does
not replace causality. Learned positional embeddings are bounded by `context_length`; oversized
sequences fail clearly rather than being silently truncated. Linear/embedding weights use a
normal initialization (default std 0.02), linear biases are zero, and LayerNorm starts with unit
scale and zero bias. Tied output embeddings share the exact parameter object.

Custom initialization explicitly zeros the token embedding's padding row after weight tying.
Embedding lookups suppress gradients for that row; a tied output head can still contribute
gradients to the shared row in future training. This initialization fix does not freeze the row.
`add_k`, `layer_norm_eps`, and `initialization_std` accept finite positive values (including values
above one); dropout and the existing ratio fields retain their original bounds.

```bash
python scripts/inspect_transformer.py --tokenizer models/tokenizer --config config/local.yaml --forward-test
python scripts/inspect_transformer.py --tokenizer models/tokenizer --config config/gpu.yaml
```

The inspector only builds and reports architecture metadata; it never trains. If no tokenizer
artifact exists, first run the Phase 4 tokenizer workflow. Weight-memory estimates cover parameter
storage only, not activations, optimizer state, or training memory. The Transformer has up to the
configured causal context (128 locally, 256 in the GPU profile), while the trigram sees exactly two
previous tokens; no comparative claim is made until later controlled experiments.

## Phase 3 preprocessing

Phase 3 consumes Phase 2 JSONL without changing it and writes a separate cleaned corpus.
The order is fixed to prevent leakage:

```text
Phase 2 JSONL → validate → normalize → quality filter → exact deduplicate → clean JSONL → canonical splits
```

Deduplication happens before split assignment, so normalized duplicate content cannot land
in both train and test. The canonical `data/splits/train.jsonl`, `validation.jsonl`, and
`test.jsonl` files will be shared unchanged by the later trigram and Transformer work.

Normalization is deliberately conservative: it applies configured Unicode normalization,
converts line endings to LF, optionally converts non-breaking spaces, removes invalid
control characters, normalizes horizontal whitespace, bounds repeated blank lines, and
trims edges. It preserves case, punctuation, numbers, URLs, emails, code fragments, file
paths, technical symbols, and paragraph boundaries. It does not remove stop words, stem,
lemmatize, or train a tokenizer.

Quality checks reject empty text, too-short text, excessive control characters, extreme
single-character repetition, and low alphabetic ratio. The defaults are permissive for technical prose. Long documents are
truncated deterministically by default; the record has `truncated: true` and its original
character count. Setting `long_document_policy: skip` writes a controlled rejection instead.

Exact duplicate detection uses the normalized UTF-8 SHA-256. Only hashes and first document
IDs are retained in memory, never the corpus text. The first accepted record in input order
wins; rejections record `DUPLICATE` and `duplicate_of` without retaining duplicate text.

Splits use a hash bucket derived from `normalized_sha256` and the global `random_seed`.
This is deterministic and streaming-friendly: no shuffle or full-text collection is needed.
Actual split counts can differ slightly from requested 80/10/10 proportions. The manifest
records the seed, requested configuration, actual counts, statistics and dataset fingerprint.

Each accepted clean JSONL record contains its Phase 2 ID/source metadata, raw and normalized
hashes, normalized text, original/final character counts, status, truncation flag, and preserved
Phase 2 metadata. Rejected records go to a separate `clean_documents.rejections.jsonl` sidecar
with no text body. The preprocessing manifest lives under `experiments/preprocessing/`.

Output files are written to unique temporary paths and renamed only after a successful run.
Existing output requires explicit `--overwrite`; an interrupted run leaves identifiable temporary
artifacts and an `interrupted` manifest. Phase 3 does not provide resume/append behavior.

### Preprocessing configuration

Both profiles provide centralized `preprocessing` and `split` sections. Local defaults use a
200,000-character maximum per document and progress every 100 records; GPU defaults raise the
document maximum to 500,000 and progress interval to 1,000. The configured working-document cap,
maximum-document cap, and absolute 100,000 homework limit still bound each run.

### Preprocessing UI and CLI

Use **Preprocessing** in Streamlit for small runs and inspection. It keeps only aggregate results,
paths, and a capped preview in session state. For large corpora, use the batch command on the
training server:

```bash
python scripts/prepare_dataset.py --help
python scripts/prepare_dataset.py \
  --input data/processed/documents.jsonl \
  --clean-output data/processed/clean_documents.jsonl \
  --split-dir data/splits \
  --config config/local.yaml
```

GPU-server preparation uses the same CPU code and does not need CUDA:

```bash
python scripts/prepare_dataset.py \
  --input data/processed/documents.jsonl \
  --clean-output data/processed/clean_documents.jsonl \
  --split-dir data/splits \
  --config config/gpu.yaml \
  --limit 100000
```

Optional `--limit` only lowers the configured cap. `--seed` deliberately overrides the global
seed for that run and is recorded in its manifest. `--no-split` writes clean JSONL only.
`--overwrite` explicitly replaces existing output files. The corpus owner is responsible for
ensuring that source material is appropriate to use; preprocessing does not provide PII/secret
detection and never logs document bodies.

## Phase 2 ingestion workflow

```text
Discover one source → validate → parse one candidate → write JSONL → update counters
                                    ↓
                           bounded preview + manifest
```

`src/data/models.py` defines typed records, options and statuses; `validators.py` guards
sources and extraction sizes; `discovery.py` walks directories; `registry.py` selects
format parsers; `ingestion.py` streams results; `statistics.py` holds counters only;
`uploads.py` stages small browser batches. `scripts/ingest_dataset.py` and the Documents
page use `ingest_directory(source_dir, options, output=..., progress=...)`.

Original files remain under `data/raw/` or your chosen external corpus directory.
Extraction never modifies them. Uploads are copied into unique `data/raw/uploads/`
subdirectories. JSONL lives under `data/processed/`; manifests default to
`experiments/ingestion/<unique-run-id>.json`. No cleaned or split dataset is produced.

### Formats and CSV behavior

- TXT: UTF-8 and UTF-8 BOM; Windows-1252 fallback is explicitly recorded as
  `encoding_fallback: true`. No replacement/ignored undecodable bytes. Binary control
  characters are rejected. Encoding detection is heuristic, not proof of the original encoding.
- PDF: PyMuPDF extracts text page by page and records page count. Blank/scanned PDFs
  become `NO_EXTRACTABLE_TEXT`; encrypted PDFs become `PASSWORD_PROTECTED`. No OCR.
- DOCX: python-docx extracts body paragraphs and tables in order, including nested
  table cells, with simple line separation. It records paragraph/table counts. Styling,
  headers/footers, tracked revisions and floating text boxes are not reconstructed.
- CSV: UTF-8/BOM, comma delimiter, quoted commas and multiline fields. `rows` creates
  one candidate per row; `file` concatenates rows into one bounded document. Select
  text columns explicitly in their desired order, e.g. `title` then `body` separated
  by a newline. Numeric strings are preserved. Empty selected values are skipped.
  Missing/duplicate headers, malformed rows, and decoding errors are reported.
  Width-mismatched rows are rejected individually; a CSV syntax/encoding failure
  terminates that CSV but ingestion continues with the next source.

The CSV reader advances one logical row at a time, which is finer-grained than chunked
DataFrames. No chunk-size setting is needed. It retains the standard 128 KiB field
ceiling to bound unusually large fields. There is no CSV dialect autodetection or
non-UTF-8 CSV fallback. A literal `null` is text; CSV has no universal null sentinel.

Parser API references: [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
and [python-docx document iteration](https://python-docx.readthedocs.io/en/latest/_modules/docx/document.html).

### Small browser uploads

Open **Documents → Upload Files**, select small samples, choose CSV mode/text columns,
then press **Ingest Dataset**. Results show actual counts, output paths, character
statistics, type distribution and a capped plain-text preview. Every UI run gets a
new JSONL filename; it does not overwrite previous results. The page also offers
**Existing Raw Folder**, using the configured `DATA_DIR/raw` only.

**Streamlit uploads are recommended for development/small datasets. Batch directory
ingestion is recommended for large datasets and the final GPU-server corpus.**
Large datasets should be placed in the configured data/raw directory and ingested
using the batch/server ingestion workflow. Do not upload 100,000 files through a browser.
Streamlit holds browser uploads in memory before application-level aggregate checks;
the 20-file/50-MiB batch cap rejects oversized selections, but cannot prevent their
initial transfer. The uploader's per-file limit is enforced by Streamlit itself.

### Limits, configuration and count semantics

Local additions (GPU changes are noted below):

```yaml
dataset:
  max_documents: 100000
  working_document_limit: 1000
  max_file_size_mb: 25
  min_extracted_characters: 20
  recursive: true
ingestion:
  output_format: jsonl
  output_path: processed/documents.jsonl
  manifest_dir: ingestion
  preview_characters: 2000
  preview_documents: 20
  max_extracted_characters: 2000000
  max_docx_uncompressed_mb: 50
  progress_interval: 100
  max_upload_files: 20
  max_upload_total_mb: 50
csv:
  mode: rows
  text_columns: []
```

This is an excerpt; retain the other Phase 1 fields. GPU defaults use 100,000 working
candidates, 100 MiB per source, 8,000,000 extracted characters, a 100 MiB DOCX expanded
size ceiling and progress every 1,000 records. Browser upload limits remain small.
`output_path` is relative to `DATA_DIR`; `manifest_dir` is relative to `EXPERIMENT_DIR`.
These paths may not contain parent traversal. CLI source/output paths resolve relative
to the project root, with absolute external paths accepted. Output/manifest locations
must be outside the selected corpus tree to avoid ingesting generated artifacts.

The effective cap is never above the smaller of the working limit, configured maximum,
and 100,000. `--limit` may only lower that cap. Every emitted candidate/rejection
consumes a slot, including unsupported files, empty rows and parse errors; this bounds
work even for a bad corpus. Therefore accepted document count can be below the cap.
CSV row candidates share the same budget with TXT/PDF/DOCX candidates. Ingestion does
not parse one extra row/file merely to determine whether more input exists.
`limit_reached` means the budget was exhausted, even if the corpus happened to end there.

Limits apply per run. Never blindly concatenate multiple runs and assume the combined
corpus remains within the homework cap; later dataset preparation must enforce its
own final corpus count. Phase 2 neither appends to existing output nor removes duplicates.

### JSONL, manifests and statuses

Each JSONL line contains `document_id` (UUID), source name/path/type, file size,
`extracted_text`, character count, extraction status/error, UTC timestamp, SHA-256,
and parser metadata. The hash covers the extracted UTF-8 text, not the original
binary source, and is calculated once per candidate. Identical text hashes remain
in the output: duplicate removal belongs to Phase 3. CSV row indexes preserve provenance.
Text with `TOO_SHORT` or no meaningful characters remains raw in its rejection record.

**Consumers must select `extraction_status == "SUCCESS"` for accepted documents.**
`records_written` / `total_documents` count all emitted records; `documents_created`
and `successful` count accepted records only. `successful + skipped + failed` equals
`records_written`. No-extractable-text and unsupported counts are subsets of skipped.
Character min/max/average cover all emitted records, including zero-length failures.
Source bytes count each examined file once, not once per CSV row; a partially read CSV
still contributes its full on-disk size. Type distributions distinguish files and records.

Statuses: `SUCCESS`, `UNSUPPORTED_TYPE`, `FILE_TOO_LARGE`, `EMPTY_FILE`,
`NO_EXTRACTABLE_TEXT`, `TOO_SHORT`, `PARSE_ERROR`, `ENCODING_ERROR`,
`PASSWORD_PROTECTED`, `UNSAFE_PATH`, `READ_ERROR`, and `EXTRACTION_TOO_LARGE`.
Limit exhaustion is a manifest flag, not a fabricated extra document.

Manifests record the run ID, timestamps, effective options, configuration, source,
output, counters and state. They contain no document text. JSONL is flushed at progress
intervals and at completion. An interrupted run retains partial output with an
`interrupted` manifest; abrupt process termination can leave a `running` manifest.
Resume/append is intentionally deferred: rerun to a new output path and inspect the
previous manifest. Existing JSONL is never overwritten.

### Batch CLI examples

Run in an activated environment from the repository root:

```bash
python scripts/ingest_dataset.py --help
python scripts/ingest_dataset.py --source data/raw --output data/processed/local-run.jsonl --config config/local.yaml --recursive --limit 100
python scripts/ingest_dataset.py --source data/raw --output data/processed/csv-run.jsonl --config config/local.yaml --csv-mode rows --text-columns title body
```

The second CSV command uses those columns for every CSV; sources missing a selected
column receive controlled errors. `--no-recursive` disables traversal. Defaults use
the configured raw directory/output location. Repeated commands need new output names.

Container execution (CPU ingestion works on any server; CUDA is unnecessary):

```bash
docker compose exec app python scripts/ingest_dataset.py --help
docker compose exec app python scripts/ingest_dataset.py --source data/raw --output data/processed/container-run.jsonl --config config/local.yaml --limit 100
```

Future GPU-server corpus, using the same Python code:

```bash
python scripts/ingest_dataset.py --source /datasets/homework3 --output data/processed/server-run.jsonl --config config/gpu.yaml --recursive --limit 100000 --csv-mode rows --text-columns text
```

For an external corpus in Docker, explicitly mount it read-only:

```bash
docker compose -f docker-compose.gpu.yml run --rm -v /datasets/homework3:/corpus:ro app python scripts/ingest_dataset.py --source /corpus --output data/processed/server-run.jsonl --config config/gpu.yaml --recursive --text-columns text
```

CLI exit codes: 0 completed (may include skipped records), 1 completed with failed
records or an I/O failure, 2 invalid request/configuration or existing output, 130 interrupted.
Logs contain periodic counts and the first 20 rejection reasons/source names, never
extracted text; the manifest/JSONL retain complete status accounting.

### Resource safeguards and limitations

Only one bounded document/CSV row and a capped preview remain in memory. Directory
names are sorted one directory at a time, not globally across the corpus. TXT files
are byte-bounded; PDF text is bounded after each page; DOCX archives are expansion-checked
before loading XML. Symlinks are rejected and directory traversal never follows them.
Use stable local corpus storage: concurrent hostile changes to paths during ingestion
are outside this filesystem trust model. Parser libraries may allocate native memory
for one source/page before the extracted-text cap is checked; these guards are not a
hard process RAM or runtime limit for adversarial PDFs. No OCR or parser worker sandbox
is introduced in this phase.

Tests create tiny TXT/CSV/PDF/DOCX fixtures dynamically, including corrupt/encrypted
files and upload traversal attempts. A lightweight 1,000-row CSV test stops at 600,
checks flushed JSONL during progress callbacks, verifies the preview cap, and checks
that traced Python memory does not grow like a retained corpus. It is not a 100K benchmark.
No datasets, binary test fixtures, tokenizer outputs, models or final experiment artifacts
should be committed. Existing runtime ignores are retained; generated JSONL is also ignored.

## Phase 4 WordPiece tokenizer

Phase 4 trains a genuine Hugging Face `tokenizers` **WordPiece** vocabulary locally. It
does not download or wrap a pretrained tokenizer. The fitting iterator reads only the
canonical Phase 3 `data/splits/train.jsonl` file in bounded JSONL batches; validation
and test splits are encoded only afterwards to report analysis. This keeps vocabulary
fitting free from validation/test leakage.

The configured special tokens (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]` by default), unknown
token, continuation prefix, vocabulary size, minimum frequency, and maximum word length
are validated in the central YAML profiles. The default tokenizer preserves case to match
the Phase 3 cleaned corpus. Encoding has no padding or truncation by default; callers can
explicitly request the configured BOS/EOS post-processing.

Train after preprocessing has created canonical splits:

```bash
python scripts/train_tokenizer.py --help
python scripts/train_tokenizer.py \
  --train data/splits/train.jsonl \
  --validation data/splits/validation.jsonl \
  --test data/splits/test.jsonl \
  --dataset-manifest experiments/preprocessing/<run>.json \
  --config config/local.yaml
```

Artifacts are constrained to `models/` (by default `models/tokenizer/`) and refuse to
replace non-placeholder content unless `--overwrite` is explicit. The output contains
`tokenizer.json`, WordPiece `vocab.txt`, `tokenizer_config.json`,
`tokenizer_manifest.json`, and `training_statistics.json`. The manifest records the
actual/requested vocabulary sizes, special-token IDs, configuration and artifact
fingerprints, train source/fingerprint when supplied, corpus document/character counts,
timestamps, duration, package/project versions, and per-split analysis. Statistics cover
documents, characters, tokens, average/min/max/median/p95 token length, unknown count/rate,
characters per token, and subword fertility.

Use the **Tokenizer** page for a small local run, a bounded ID-ordered vocabulary preview,
and encode/decode inspection. For a final corpus, use the CLI on the training server; the
WordPiece fit is CPU-oriented and the UI never retains the corpus in session state. A missing
split or malformed JSONL produces a controlled error rather than silently fitting a different
dataset. Phase 4 creates no neural model, embeddings, checkpoints, trigram model, generation,
or training/evaluation metrics.

## Phase 7 Transformer training engine

The reusable engine in `src/training/` trains the existing custom Phase 6 model;
there are no pretrained weights, external trainers, or distributed dependencies.
`scripts/train_transformer.py` and the Training page call the same service.

### Objective and bounded data pipeline

Training accepts only canonical `data/splits/train.jsonl`,
`data/splits/validation.jsonl`, and `models/tokenizer/` (or their configured runtime
root equivalents). It never opens the test split. The actual WordPiece vocabulary,
special-token IDs, tokenizer JSON SHA-256, train/validation SHA-256, and architecture
fingerprint are checked and recorded. Newly generated preprocessing manifests include
split hashes, and newly fitted tokenizer manifests include the training-source hash.
An optional `--dataset-manifest experiments/preprocessing/<run>.json` checks this
provenance. Legacy tokenizer manifests without the source hash produce a warning:
the current input bytes are fingerprinted, but historical fitting provenance cannot
be reconstructed. A declared whole-corpus fingerprint is not a substitute for the
actual train/validation hashes. Resume requires matching fingerprints and vocabulary.

Each document becomes `[BOS] tokens [EOS]`, independently of every other document.
A window of up to `context_length + 1` source tokens produces `window[:-1]` inputs
and `window[1:]` targets. Thus context length is the maximum **input** length, not
the source-window length. Short final chunks are retained; there is no cross-document
packing or invisible transition. `sequence_stride <= context_length` permits overlap;
the defaults use full-context strides. Validation uses full-context strides and no shuffle.
The collator pads inputs with the actual PAD ID, targets with `-100`, and returns a
boolean real-position mask. Causal attention and padding attention remain separate;
cross-entropy consumes raw logits and ignores only padded targets.

The IterableDataset tokenizes one document at a time. Workers partition line indices
by `line_index % num_workers`; each scans the file but only parses/tokenizes its own
records. A deterministic, epoch-seeded bounded shuffle buffer holds sequences, not the
whole corpus. Memory therefore includes one document per worker plus each worker's
buffer, prefetched batches, model, gradients, and optimizer states. An unusually large
single document can still consume significant RAM. Spawn workers, one-batch prefetch,
shared epoch state for persistent workers, and explicit worker shutdown are supported.
Validation is ordered with zero workers. Local defaults use one worker and a 1,000-item
buffer; GPU defaults use four workers, pinned memory, persistence, and 10,000 items
**per worker**. Reduce these values if host memory is constrained.

### Updates, precision, and recovery

AdamW excludes biases and LayerNorm vectors from weight decay and counts tied weights
once. Linear warmup followed by cosine decay is indexed by optimizer updates, not
microbatches. Warmup is capped to the schedule horizon. With `max_steps: null`, a
streaming count-only pass determines the horizon without retaining tokenized data.
With a step limit, that limit supplies the horizon. Limits are total optimizer steps,
including already completed steps when resuming; the epoch limit also applies.

Backward accumulates **summed** token losses. At the update boundary, FP16 gradients
are unscaled, divided by the actual accumulated valid-target count, clipped with
`clip_grad_norm_`, and stepped. The final partial group uses its actual token count,
so neither padding nor shorter batches distort the objective or discard gradients.
Nominal effective single-GPU sequence batch = microbatch size × accumulation steps;
partial groups can be smaller. Training and validation NLL are token-weighted;
monitoring perplexity is `exp(NLL)`, or infinity on overflow.

CPU uses FP32; explicit CPU FP16/BF16 is rejected. CUDA `precision: auto` prefers
PyTorch-reported BF16 support and otherwise chooses FP16. Explicit unsupported BF16
fails clearly. FP16 uses modern autocast and GradScaler; BF16 uses autocast without
a scaler. `mixed_precision: false` selects FP32. `DEVICE=cuda` requires CUDA and
never silently falls back to CPU. Actual precision and environment are recorded.
Python, NumPy, CPU and CUDA RNGs are seeded. Deterministic mode can reduce performance;
bit-identical results across hardware/PyTorch versions are not promised.

Validation runs without gradients at epoch ends and step-limit stops. Lowest validation
NLL selects `best.pt`; early stopping separately applies `min_delta` and `patience`.
Checkpoints contain model, AdamW, schedule, optional scaler, counters, early-stopping
state, configuration, fingerprints, and RNGs. Loading uses `weights_only=True` on
project-generated state dictionaries, never whole model objects. Resume reconstructs
the deterministic stream and skips previously consumed batches, then restores the
saved RNG state. This costs replay time; keep worker/data/batch/shuffle settings fixed.
Only epoch and step limits may change. Increasing a limit does **not** restart or
stretch the saved LR schedule: beyond its original horizon the minimum LR applies.
Increase `--epochs` as well if the saved run has exhausted its epoch allowance.

Each checkpoint file is staged and atomically replaced; archive retention never
prunes `best.pt` or `latest.pt`. Ctrl+C records `INTERRUPTED` and, when safe, saves
the last complete optimizer boundary; partial gradients are discarded and replayed.
An interruption inside an optimizer mutation relies on the previous valid checkpoint.
OOM records `FAILED_OOM`; NaN/Inf loss or gradients record `FAILED_NONFINITE` and
stop before publishing a bad best export. Other statuses are `RUNNING`, `COMPLETED`,
`EARLY_STOPPED`, `FAILED_CONFIGURATION`, and `FAILED_OTHER`. There is no automatic
OOM retry loop: reduce microbatch size/context, then use accumulation to recover the
desired effective batch. Previous valid checkpoints/exports are preserved on failure.

### Runtime artifacts

```text
experiments/training/<run_id>/
  resolved_config.json  environment.json  history.csv  training.log  summary.json
checkpoints/<run_id>/
  latest.pt  best.pt  checkpoint_step_*.pt  interrupt.pt (when interrupted safely)
models/transformer/
  best_model.pt  model_manifest.json
```

History is appended incrementally, with optimizer steps, losses, throughput, LR,
gradient norm, RSS, and CUDA allocated/reserved/peak memory where available. Training
elapsed time begins before batch processing and includes training/validation; it
excludes ingestion, preprocessing, tokenizer fitting, and initial setup. The overall
attempt duration in the summary also includes final export work.

Successful training exports the best validation **state_dict**, not necessarily the
latest weights. Its manifest records architecture, vocabulary, fingerprints, parameter
count, precision, best epoch/step/NLL/perplexity, timing, tokens, size, and environment.
Existing exports require `--overwrite-export`. Model/manifest staging and backup
rollback protect against caught write failures. Atomicity is per file, not a power-loss
transaction across multiple files: use one writer per run/export directory and retain
checkpoints for recovery. Do not manually edit checkpoint contents.

### CLI: local checks and resume

After preparing canonical splits and fitting WordPiece, use the local profile first.
These commands are examples for your runtime data, not a claim of a final corpus run.

```bash
python scripts/train_transformer.py --help
DEVICE=cpu python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/local.yaml --dry-run
DEVICE=cpu python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/local.yaml --max-steps 5
DEVICE=cpu python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/local.yaml \
  --resume checkpoints/<run_id>/latest.pt --max-steps 10 --epochs 4 --overwrite-export
```

Dry run performs one forward/loss/backward/gradient check and writes diagnostic
metadata, but takes no optimizer step and writes no model/checkpoint export.
`--output` selects a run-container directory beneath the configured experiment root;
`--run-name` is a label, not an arbitrary filesystem path. Real CLI runs validate the
entire validation split, even when optimizer steps are bounded.

The Streamlit Training page shows environment/configuration, runs a dry check or a
synchronous small run (default 5, hard maximum 100 optimizer steps), and lists recent
run summaries without loading entire histories. It additionally caps local model size,
microbatch/accumulation, shuffle, workers, and validation to two batches. Such UI
validation is explicitly marked as a subset in metadata and is not a final benchmark.
There are no detached training jobs. Use the CLI for server work.

### GPU server workflow (future execution)

Obtain a fresh clone under your own Git control; all engine source, dependencies,
profiles, scripts and tests belong in the repository. Create runtime directories as
described above, verify the host NVIDIA driver (`nvidia-smi`), Docker, and NVIDIA
Container Toolkit, then build the GPU image. The host driver must support the CUDA
runtime selected by the image (see the earlier GPU compatibility section). Do not
install host drivers inside the container. CPU builds use the CPU wheel index;
`docker-compose.gpu.yml` selects CUDA 12.8 wheels and requests one GPU on service `app`.

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm app \
  python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Provide the raw corpus, then run the documented Phase 2 ingestion, Phase 3 preparation,
Phase 4 tokenizer, Phase 5 trigram, and Phase 6 architecture-inspection commands in
that order. Keep the same canonical split/tokenizer artifacts for the comparison.
Next run only the following dry check and a short sanity run:

```bash
DEVICE=cuda python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/gpu.yaml --dry-run
docker compose -f docker-compose.gpu.yml run --rm app \
  python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/gpu.yaml --dry-run
```

Replace `--dry-run` with `--max-steps 5` for a bounded real GPU check. The eventual
full command below is **reserved for Phase 9**, after Phase 8 shared evaluation;
do not execute it as a Phase 7 check:

```bash
DEVICE=cuda python scripts/train_transformer.py \
  --train data/splits/train.jsonl --validation data/splits/validation.jsonl \
  --tokenizer models/tokenizer --config config/gpu.yaml --output experiments/training
```

GPU code is implemented, but CUDA runtime execution must be verified on the actual
server; CPU tests and mocked precision tests do not establish GPU success or throughput.
No final test evaluation, Transformer generation framework, or 100K experiment belongs
to Phase 7. Do not commit datasets, split JSONL, tokenizer vocabulary/artifacts, trigram
databases, model weights, checkpoints/optimizer states, histories/logs/manifests,
TensorBoard/profiler output, `.env`, or secrets. Commit source/config/tests/docs only;
runtime directories remain ignored. No local-only notebook/helper is required.

## Phase 8 shared evaluation and comparison

The evaluation framework is implemented; **final full-corpus results are not yet
available**. Phase 9 remains a separate, user-controlled experiment. Evaluation never
trains either model, fits WordPiece, updates counts/weights, chooses checkpoints,
or changes the test split. Temporary synthetic test fixtures create tiny models only
to test the framework; no production artifacts are required for the test suite.

### Fairness and exact prediction events

Both adapters read the same canonical `DATA_DIR/splits/test.jsonl`, in the same order,
using the same canonical WordPiece tokenizer, vocabulary, and document limit. A
shared streaming iterator holds one document at a time; the Transformer adapter
batches only a bounded number of causal windows. There is no full-corpus token list.

The common targets for document `A B C` are **A, B, C, EOS**. Trigram uses histories
`BOS,BOS → A`, `BOS,A → B`, `A,B → C`, `B,C → EOS`; its two BOS history tokens create
no extra target. Transformer uses the Phase 7 causal shift from `BOS A B C EOS`.
Empty documents contribute the single EOS event. Documents never share context.
Literal PAD in test text is rejected rather than silently changing the denominator.

Transformer scoring always uses `stride = context_length`, irrespective of training
overlap. Each window consumes up to context+1 source IDs, and its labels are scored
exactly once. Adjacent windows share a boundary input token but **not a target event**.
Long documents reset learned positions at each window and lose earlier context at
that boundary. This matches the Phase 7 non-overlapping validation policy; it is not
a maximum-context sliding-window likelihood estimator. Trigram keeps its two-token
history throughout each document. These designed context differences are disclosed,
not disguised by restricting the Transformer to two tokens.

```text
Trigram:      P(t_i | t_(i-2), t_(i-1))
Transformer:  P(t_i | preceding tokens within the active context window)
Average NLL:  -1/N × sum(log P(t_i | context))
Perplexity:   exp(average NLL)
Bits/token:   average NLL / ln(2)
```

Natural logarithms and exact valid-target counts are used. Matching WordPiece makes
token-level NLL/perplexity substantially more defensible than comparing different
tokenizations, but does not make all architectural conditions identical. No test
hyperparameter tuning or model selection is performed. The software does not assume
the Transformer wins; factual summaries reflect the measured values.

### Identity and read-only guarantees

The canonical tokenizer JSON hash must match both model manifests; actual vocabulary,
special IDs, SQLite metadata, and Transformer architecture are checked. Available
corpus/training-source hashes must agree. `--dataset-manifest` checks a completed
preprocessing manifest, including its test/train/validation hashes where supplied.
A changed historical test split fails when its recorded hash is available. Without
that manifest/hash, a warning explicitly states that historical test identity cannot
be proven; hashing current bytes cannot recover missing historical provenance.

The actual test, tokenizer, prompts, model files and manifests are hashed before and
after evaluation. Changed inputs fail publication of a complete comparison. Both
adapters must report identical documents/events/test/tokenizer identities before a
primary comparison is emitted. This is a stable-input workflow, not a snapshot or
concurrent-writer system: do not replace artifacts during evaluation.

SQLite is opened using the Phase 5 read-only loader and queries counts directly;
it never materializes all n-gram tables. Transformer loads only its exported
`best_model.pt` state dictionary using `weights_only=True`, validates the canonical
architecture, disables parameter gradients, and uses `eval()`/`inference_mode()`.
No optimizer or training service is imported by the evaluation engine.

### Metrics and timing definitions

Results record documents, targets, total log-likelihood/negative log-likelihood,
average NLL, perplexity, bits/token, scoring duration, target throughput, actual
device/precision, model identity, and model size. Overflow returns infinity in the
Python metric and the explicit string `"inf"` in portable JSON (JSON has no infinity
number). Missing metrics remain `null` with reasons, not guessed zeroes.

Model size is the deployed SQLite database versus exported Transformer state_dict,
excluding optimizer/checkpoint/history files. Manifest size is excluded for both.
Transformer parameters are counted from the instantiated architecture; distinct
unigram/bigram/trigram counts are separately labeled, not called parameters.
Training duration and explicitly available training-resource fields are read from
existing manifests; there is no retraining to measure them. Earlier manifests may
lack peak training RAM/VRAM or throughput; these remain unavailable. RAM and VRAM
are separate metrics, never combined. Trigram recorded time includes its established
count/build/persistence work; Transformer recorded time includes training/validation
after initial setup. Neither includes preprocessing or tokenizer fitting.

Scoring uses `perf_counter` and includes streaming/tokenization/collation/model
scoring, but excludes model loading, fingerprint scans, warmup and export. CUDA is
synchronized around the measured scoring section. Optional CUDA warmup replays only
the first bounded batch without counting it; the measured pass restarts and scores
every target. CPU uses FP32; CUDA precision reuses Phase 7 auto/BF16/FP16 selection.
Actual precision and warmup count are recorded. Lower the configurable evaluation
batch size on OOM; the engine fails rather than skipping examples or silently retrying.

Relative results are Transformer/trigram training-time, model-size and scoring
throughput ratios, plus `(trigram PPL - Transformer PPL) / trigram PPL × 100`.
Zero, null and nonfinite denominators return unavailable ratios. Hardware, software,
CPU/GPU availability and system RAM are recorded. CPU trigram versus GPU Transformer
timings describe practical execution cost, **not architecture-only speed**.

### Generation and human review

`config/evaluation_prompts.yaml` contains short original/general prompts; use
`--prompts` for an explicitly chosen alternate file. Both models receive identical
prompt text/WordPiece IDs, token limits, strategy and seed. Primary generation is
greedy; sampling is secondary and is not expected to produce equivalent stochastic
trajectories across models. The existing trigram generator retains its observed
candidate/unigram fallback policy; it is not full-vocabulary Lidstone sampling.

Transformer generation takes the last-position logits, stops at EOS or the configured
token limit, and crops to the most recent context-length tokens at every step. Older
history leaves attention; positions restart in the cropped window. No KV cache is
implemented, so context is recomputed per token. PAD/BOS are excluded from Transformer
generation and special tokens are hidden by canonical decoding. Continuation text
does not include the original prompt. Generated-token counts exclude stopping EOS.
Generation-only timing excludes loading and prompt tokenization and synchronizes CUDA
around the loop. Small host/device synchronization for token selection is inherent
in this simple implementation; these are not optimized production inference numbers.

Each output includes prompt-token count, continuation IDs/text, token count, duration,
throughput, EOS flag, strategy and seed. A human rubric provides blank ratings/notes
for coherence, relevance, non-repetition, continuity, completeness and plausibility.
The scale is 1 (poor)–5 (strong); no scores or subjective superiority are fabricated.
There is no external LLM judge or automatic human-rating persistence.

### CLI and Docker

Local **development subset**, after canonical model artifacts exist:

```bash
python scripts/evaluate_models.py --help
DEVICE=cpu python scripts/evaluate_models.py \
  --test data/splits/test.jsonl --tokenizer models/tokenizer \
  --trigram-model models/trigram --transformer-model models/transformer \
  --config config/local.yaml --limit-documents 100 --output experiments/evaluation
```

Add `--dataset-manifest experiments/preprocessing/<run>.json` to verify historical
split hashes. `--model trigram` or `--model transformer` evaluates only that artifact;
default `both` requires both and does not silently downgrade. Local YAML defaults to
100 documents; GPU YAML has no limit. `--full-test` explicitly removes a profile
limit and cannot be combined with `--limit-documents`. Any configured limit is labeled
**DEVELOPMENT SUBSET**, even if a tiny file happens to end before that limit.

Future full evaluation, **reserved for Phase 9; not executed in Phase 8**:

```bash
DEVICE=cuda python scripts/evaluate_models.py \
  --test data/splits/test.jsonl --tokenizer models/tokenizer \
  --trigram-model models/trigram --transformer-model models/transformer \
  --config config/gpu.yaml --output experiments/evaluation
docker compose -f docker-compose.gpu.yml run --rm app \
  python scripts/evaluate_models.py \
  --test data/splits/test.jsonl --tokenizer models/tokenizer \
  --trigram-model models/trigram --transformer-model models/transformer \
  --config config/gpu.yaml --output experiments/evaluation
```

The existing GPU Docker service is `app`; its CUDA-wheel/host-driver/Container Toolkit
requirements are unchanged. A fresh source distribution contains all evaluation
logic and prompts. No new heavy dependency, notebook, local helper, or external
service is required. CUDA evaluation must still be exercised on a real GPU server;
CPU tests and synchronization mocks are not GPU runtime validation.

### UI, exports, and failures

Evaluation runs synchronously with a default of 10 documents and a hard UI cap of
100, batch size at most four, generation at most 20 tokens per prompt, and small-model
architecture checks. Long documents still cost time: use small data and the CLI for
server jobs. No detached job manager exists. Comparison and Generate Text display
completed runs, validity/provenance warnings, the metric table, relative values,
factual interpretation, continuations and blank human-review rubric. JSON/CSV are
downloadable. The recent-run viewer lists at most 20 summaries. Incomplete runs are
not rendered as valid comparisons; final benchmark availability is never inferred
from framework implementation.

```text
experiments/evaluation/<evaluation_id>/
  resolved_config.json   environment.json   summary.json
  trigram_metrics.json   transformer_metrics.json
  generation_results.json   comparison.json   comparison.csv
```

Artifacts carry a common evaluation ID and reproducibility hashes. Config records
the actual limit, generation settings, prompts and seed. CSV provides metric, both
model values and unit, with identity columns. JSON includes metadata, missing-value
reasons, relative comparisons and environment. Writes use same-directory staging and
atomic per-file replacement. The unique run's `summary.json` is authoritative:
`RUNNING`, `COMPLETED`, `FAILED_CONFIGURATION`, `FAILED_FINGERPRINT`,
`FAILED_MODEL_LOAD`, `FAILED_EVALUATION`, or `INTERRUPTED`. Single-model metrics from
an incomplete attempt are labeled partial. There is no whole-directory power-loss
transaction; retain the summary and do not treat orphaned outputs as final results.

Commit only source, configurations/prompts, tests and documentation under your own
Git control. Do not commit the test split, tokenizer, SQLite model, Transformer
weights/checkpoints, runtime evaluation/comparison CSV/JSON, generated continuations,
training histories, final results, `.env`, or secrets. No Phase 9 experiment is
automatically launched by Phase 8.
