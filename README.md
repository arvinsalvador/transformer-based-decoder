# Transformer-Based Decoder-Only Language Model

University machine-learning homework. **Status: Phase 1 — foundation and environment only.**
The dashboard and environment diagnostics work. No document parsing, tokenization,
model architecture, training, generation, or evaluation is implemented yet.

## Homework requirements

- Train a decoder-only Transformer using a WordPiece tokenizer.
- Use a chosen topic with **at most 100,000 documents**. This is a maximum, not a minimum.
- Compare model performance and training time against a trigram language model.
- Start development with small subsets; use the GPU profile for final experiments later.

Document count alone does not describe dataset size. Token counts will be measured
in later phases. The local profile selects 1,000 documents but Phase 1 loads none.
The random seed is recorded for later splitting and training; it is not applied to
any simulated training process.

## Architecture and directories

The UI calls centralized configuration and device utilities. Future computational
modules are empty packages; they must remain independent of Streamlit.

```text
app/                  Streamlit entry point, dashboard component, placeholder pages
src/config/           YAML loading, validation, environment and path resolution
src/utils/            Structured CPU/CUDA diagnostics
src/data/             Phase 2–3 ingestion and preparation placeholder
src/tokenizer/        Phase 4 WordPiece placeholder
src/trigram/          Phase 5 baseline placeholder
src/transformer/      Phase 6 decoder placeholder
src/training/         Phase 7 training and checkpoint placeholder
src/evaluation/       Phase 8 evaluation placeholder
config/               local.yaml and gpu.yaml
scripts/              Reserved for later command-line utilities
tests/                Configuration, device and Streamlit tests
data/raw/             Private source documents (TXT/CSV/DOCX/PDF in later phases)
data/processed/       Generated cleaned data
data/splits/          Generated dataset splits
models/tokenizer/     Generated tokenizer files
models/trigram/       Generated baseline artifacts
models/transformer/   Generated weights
checkpoints/          Future resumable training state
experiments/          Future experiment metadata, metrics and logs
reports/              Generated reports
```

Runtime directories contain only `.gitkeep` placeholders initially. Store datasets,
uploads, tokenizer outputs, weights, checkpoints, logs, and reports in these ignored
locations. Large generated artifacts are excluded from version control because they
are expensive to transfer, may contain private documents, and are reproduced from
code/configuration. Retain data provenance and experiment recipes as reviewed source
configuration in later phases. Never commit `.env`, credentials, caches or virtual
environments. Filenames from future uploads must be treated as untrusted; Phase 1
accepts no uploads, executes no shell inputs and introduces no pickle loading.

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
environment manifest for final controlled experiments. Parsing/tokenizer packages are
deferred until their phases so the initial environment remains smaller.

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
reports persistently. It launches only the Phase 1 dashboard. Training commands will be
added in Phase 7. Use an SSH tunnel for a remote dashboard:
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
docker build --target test -t homework3-phase1-test .
docker run --rm homework3-phase1-test
docker run --rm homework3-phase1-test python -m ruff check .
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
  reduce future batch/context settings. Phase 1 never loads corpora or creates tensors.
- Dependency download failure: verify access to PyPI and download.pytorch.org, then
  retry the build. Do not substitute a CUDA build on a low-memory local machine.

## Remaining phases

2. TXT/CSV/DOCX/PDF ingestion.
3. Cleaning, preparation, dataset splits and token-size accounting.
4. WordPiece tokenizer.
5. Trigram baseline.
6. Decoder-only Transformer.
7. Training, resource controls, mixed precision and checkpoints.
8. Evaluation framework.
9. Controlled final experiments and comparisons.
10. Final UI, reporting and validation.

Phase 1 stops at the foundation. No training results or fabricated metrics are present.
