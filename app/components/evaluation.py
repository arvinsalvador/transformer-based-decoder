"""Bounded synchronous evaluation and read-only comparison viewer."""

from copy import deepcopy
from dataclasses import replace
from heapq import nlargest

import streamlit as st

from src.evaluation.identity import read_json
from src.evaluation.service import EvaluationFailure, evaluate_models


def render_comparison(settings):
    st.title("Trigram vs Transformer")
    st.info("Shared comparison framework implemented; final Phase 9 results are not implied.")
    root = settings.paths["EXPERIMENT_DIR"] / "evaluation"
    paths = nlargest(20, root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        st.write("No evaluation runs available. Use Evaluation for a development check.")
        return
    selected = st.selectbox("Evaluation run", paths, format_func=lambda p: p.parent.name)
    try:
        summary = read_json(selected)
        st.json(summary)
        if summary["status"] != "COMPLETED":
            st.warning("Incomplete run: partial metrics are not a valid comparison.")
            return
        result = read_json(selected.parent / "comparison.json")
        st.subheader(result["scope"])
        st.json(result["fingerprints"])
        st.dataframe(result["table"], hide_index=True)
        st.json(result["relative"])
        for statement in result["interpretation"]:
            st.write(statement)
        for item in read_json(selected.parent / "generation_results.json")["results"]:
            with st.expander(f"{item['model_type']} — {item['prompt']}"):
                st.text(item["generated_text"])
                st.json(item)
        st.caption("Human-review ratings remain blank; no automatic judge or fabricated ratings.")
        for filename in ("comparison.json", "comparison.csv"):
            st.download_button(
                f"Download {filename}", (selected.parent / filename).read_bytes(), filename
            )
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot read evaluation artifacts: {exc}")


def render_evaluation(settings):
    st.title("Model Evaluation")
    st.info(
        "Read-only evaluation. Browser runs are DEVELOPMENT SUBSET checks, never final benchmarks."
    )
    root = settings.paths["MODEL_DIR"]
    test = settings.paths["DATA_DIR"] / "splits/test.jsonl"
    files = [
        test,
        root / "tokenizer/tokenizer.json",
        root / "trigram/trigram_counts.sqlite",
        root / "transformer/best_model.pt",
    ]
    st.write({str(path): "Available" if path.is_file() else "Missing" for path in files})
    st.json(settings.values["evaluation"])
    st.caption(
        "Fingerprints are validated before scoring; historical test verification "
        "needs a preprocessing manifest in the CLI."
    )
    limit = st.number_input("Development test document limit", min_value=1, max_value=100, value=10)
    if st.button("Run Evaluation", disabled=not all(path.is_file() for path in files)):
        try:
            architecture = read_json(root / "transformer/model_manifest.json")["architecture"]
            if any(
                architecture[key] > maximum
                for key, maximum in (
                    ("context_length", 256),
                    ("embedding_dim", 256),
                    ("num_layers", 4),
                    ("feedforward_dim", 1024),
                )
            ):
                raise ValueError("Model exceeds browser safety limits. Use the CLI.")
            values = deepcopy(settings.values)
            values["evaluation"]["gpu_warmup_batches"] = min(
                1, values["evaluation"]["gpu_warmup_batches"]
            )
            values["evaluation"]["transformer_batch_size"] = min(
                4, values["evaluation"]["transformer_batch_size"]
            )
            values["evaluation"]["generation"]["max_new_tokens"] = min(
                20, values["evaluation"]["generation"]["max_new_tokens"]
            )
            with st.spinner("Scoring the shared development subset..."):
                summary = evaluate_models(
                    replace(settings, values=values),
                    test,
                    root / "tokenizer",
                    limit_documents=int(limit),
                )
            st.json(summary)
            st.success(
                "Evaluation completed. Open Comparison to inspect metrics and continuations."
            )
        except (EvaluationFailure, ValueError, OSError, KeyError) as exc:
            st.error(str(exc))
    st.code(
        "python scripts/evaluate_models.py --test data/splits/test.jsonl "
        "--tokenizer models/tokenizer --config config/local.yaml --limit-documents 100"
    )
