# GPU server checklist

- [ ] Obtain the source under your own Git control; no runtime artifacts in source.
- [ ] NVIDIA host driver and `nvidia-smi` work.
- [ ] Docker and NVIDIA Container Toolkit expose a GPU to containers.
- [ ] Build the current GPU image; verify PyTorch CUDA runtime and GPU visibility.
- [ ] Runtime volumes exist and are writable; disk satisfies the plan's safety minimum.
- [ ] Supply raw corpus, complete ingestion/preprocessing and verify ≤100K TOTAL documents.
- [ ] Keep completed preprocessing manifest and split hashes.
- [ ] Fit canonical WordPiece on train only; retain vocabulary/fitting provenance.
- [ ] Plan and static preflight pass; separate GPU dry run passes.
- [ ] Execute a selected small scale; inspect status, artifacts and checkpoint/resume.
- [ ] Explicitly confirm FULL only after reviewing configuration and available resources.
- [ ] FULL trigram/Transformer and shared full-test evaluation complete.
- [ ] Generate the final report; resolve integrity failures and review warnings.
- [ ] Back up data, models, checkpoints, hashes and reports outside source control.

Commands are in GPU_RUNBOOK.md. No checkbox implies an operation was executed here.
