"""Inspection only: never launches training or detached jobs."""

from heapq import nlargest

import streamlit as st

from src.config.settings import PROJECT_ROOT
from src.evaluation.identity import read_json
from src.experiments.plan import load_plan


def render_experiments(settings):
    st.title("Controlled Experiments")
    st.info(
        "Experiment orchestration implemented. Inspection only; long training runs require the CLI."
    )
    st.caption("100,000 is the TOTAL corpus cap. FULL means the actual canonical training split.")
    st.subheader("Predeclared plan")
    try:
        st.json(load_plan(PROJECT_ROOT / "config/experiments.yaml"))
        root = settings.paths["EXPERIMENT_DIR"] / "final"
        paths = nlargest(20, root.glob("*/status.json"), key=lambda p: p.stat().st_mtime)
        if paths:
            selected = st.selectbox("Experiment", paths, format_func=lambda p: p.parent.name)
            status = read_json(selected)
            st.json(
                {
                    key: status.get(key)
                    for key in ("experiment_id", "status", "plan_fingerprint", "audit", "message")
                }
            )
            for filename in ("preflight.json", "environment.json"):
                if (selected.parent / filename).exists():
                    with st.expander(filename):
                        st.json(read_json(selected.parent / filename))
            matrix = selected.parent / "experiment_matrix.json"
            if matrix.exists():
                st.dataframe(read_json(matrix)["rows"], hide_index=True)
            summary = selected.parent / "summary.json"
            if summary.exists():
                st.write(
                    {"FULL primary result available": read_json(summary)["primary_final_result"]}
                )
        else:
            st.write("No experiment runs recorded. No final results are implied.")
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot inspect experiments: {exc}")
    st.subheader("Server commands")
    st.code(
        "python scripts/run_experiments.py --config config/gpu.yaml "
        "--experiment-config config/experiments.yaml --plan\n"
        "# Then --preflight, --dry-run-only, and explicitly selected --execute stages."
    )
