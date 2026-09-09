"""Manifest-only presentation view; never loads corpus, weights or SQLite."""

import streamlit as st

from src.evaluation.identity import read_json


def report_preview(settings):
    path = settings.paths["REPORT_DIR"] / "final_summary.json"
    if not path.exists():
        return None
    return read_json(path)


def render_final_results(settings):
    st.title("Final Results")
    st.info(
        "FULL is the primary result. This view reads saved audited reports; "
        "it never trains or regenerates text."
    )
    presentation = st.checkbox("Presentation view", value=False)
    try:
        report = report_preview(settings)
        if report is None:
            st.warning("FINAL EXPERIMENT NOT YET EXECUTED or report not generated.")
            st.code(
                "python scripts/generate_final_report.py "
                "--config config/local.yaml --readiness-only"
            )
            return
        st.caption(
            f"Saved audit: {report['generated_at']}. Rerun reporting after changing artifacts."
        )
        st.write(report["project_status"])
        if report["audit_status"] == "FAIL":
            st.error("Integrity audit failed. Do not present this as a validated final result.")
            return
        st.dataframe(report["compliance"], hide_index=True)
        final = report["final"]
        if final["status"] != "PASS":
            st.warning(final.get("message", "FINAL EXPERIMENT NOT YET EXECUTED"))
            return
        st.write(
            {
                "Primary scale": "FULL",
                "Experiment": final["experiment_id"],
                "Documents": final["counts"],
            }
        )
        st.dataframe(report["comparison_table"], hide_index=True)
        for statement in final["interpretation"]:
            st.write(statement)
        for warning in final["warnings"]:
            st.warning(warning)
        st.dataframe(
            [
                {key: row[key] for key in ("scale", "status", "train_documents")}
                for row in final["learning_curve"]
            ],
            hide_index=True,
        )
        for metric in (
            "perplexity",
            "training_duration_seconds",
            "model_size_bytes",
            "tokens_per_second",
        ):
            rows = [
                {
                    "train_documents": row["train_documents"],
                    **{name: result.get(metric) for name, result in row["results"].items()},
                }
                for row in final["learning_curve"]
                if row.get("results")
            ]
            if rows:
                st.subheader(metric)
                st.line_chart(rows, x="train_documents", y=["trigram", "transformer"])
        memory = [
            {
                "train_documents": row["train_documents"],
                "transformer_peak_vram_mib": row["training_resources"]["gpu_peak_mb"],
            }
            for row in final["learning_curve"]
            if row.get("training_resources", {}).get("gpu_peak_mb") is not None
        ]
        if memory:
            st.subheader("Transformer peak allocated VRAM (MiB)")
            st.line_chart(memory, x="train_documents", y="transformer_peak_vram_mib")
        st.subheader("Saved generation examples")
        for item in final["generation"]:
            st.write(f"{item['model_type']} — {item['prompt']}")
            st.text(item["generated_text"])
        st.subheader("Limitations")
        for limitation in report["limitations"]:
            st.write(limitation)
        if not presentation:
            with st.expander("Reproducibility and audit detail"):
                st.json(final["fingerprints"])
                st.json(report["system_audit"]["findings"])
        for filename in (
            "final_report.md",
            "final_summary.json",
            "final_comparison.csv",
            "final_report.html",
        ):
            path = settings.paths["REPORT_DIR"] / filename
            if path.exists():
                st.download_button(f"Download {filename}", path.read_bytes(), filename)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot read final report: {exc}")
