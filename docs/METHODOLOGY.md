# Experimental methodology

The homework cap is 100,000 TOTAL clean documents. At 80/10/10, a 100K corpus gives
approximately 80K train, 10K validation and 10K test; FULL is the actual train count.
Hash-bucket splits approximate configured ratios; report actual counts, not assumptions.

Deduplication precedes splitting. WordPiece is fitted only on canonical train.
Both learners use the same training subset and vocabulary. Validation selects neural
best weights/early stopping. Test likelihood never controls the training process or
technical scale-progression gates. The experimenter must separately attest that test
results were not used for manual hyperparameter tuning; software cannot prove unseen work.

Trigram computes P(t_i | t_(i-2), t_(i-1)); Transformer computes causal probabilities
within its active context. Targets are document tokens plus EOS, without extra BOS
predictions. Transformer test windows use full-context strides, scoring each target
once; positions/context reset across windows. This is not maximal-context sliding
evaluation. Average NLL = -sum(log p)/N; perplexity = exp(NLL). Shared WordPiece makes
token comparisons defensible but does not erase architectural context differences.

Report training time from historical manifests, deployed model size without optimizer
states, evaluation duration/throughput and available RAM/VRAM separately. CPU-vs-GPU
timings describe practical execution cost, not hardware-independent architecture speed.
Final RSS is not peak RAM. Never invent missing resource measurements.

The scale study uses deterministic nested train subsets, fixed validation/test,
fixed train-fitted tokenizer, architecture, seeds and optimizer policy. It changes
model-training scale, not end-to-end tokenizer training scale. One seed gives no
multi-seed uncertainty estimate. FULL remains primary even when another scale looks better.
Saved prompts/outputs and blank or actual human notes are reported without cherry-picking.

## Current raw package (not a completed experiment)

The supplied SUROY README describes a synthetic tourism corpus augmented from curated
seed summaries. Do not describe it as real scraped social-media posts. If this package
is used, disclose synthetic provenance, template/domain overlap and limited external
generalization. Exact deduplication does not guarantee absence of near-duplicate
synthetic templates across splits. Canonical preprocessing must establish actual counts;
the archive's advertised 100K rows do not prove runtime document-cap compliance.
