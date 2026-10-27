"""Streamlit evaluation frontend for the saved telecom churn pipelines."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_preprocessing import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET, prepare_frame
from src.evaluation import comparison_row, score_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="Churn Signal Room", page_icon="📡", layout="wide")


@st.cache_resource
def load_artifacts():
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: __import__("joblib").load(MODEL_DIR / filename) for name, filename in metadata["artifact_files"].items()}
    return metadata, models


def validate_evaluation_frame(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame | None, str | None]:
    required = set(metadata["input_columns"]) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)
    labels = set(frame[TARGET].dropna().astype(str).unique())
    if not labels.issubset({NEGATIVE_CLASS, POSITIVE_CLASS}) or not labels:
        return None, "Churn must contain only 'No' and 'Yes' values."
    prepared = prepare_frame(frame)
    return prepared, None


def evaluate_all(models: dict, frame: pd.DataFrame, metadata: dict) -> dict:
    features = frame[metadata["input_columns"]]
    target = frame[TARGET].map(metadata["target_mapping"])
    return {name: score_model(model, features, target) for name, model in models.items()}


def show_results(results: dict, selected_name: str) -> None:
    rows = [comparison_row(name, result) for name, result in results.items()]
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({column: "{:.3f}" for column in table.columns[1:]}), use_container_width=True, hide_index=True)
    selected = results[selected_name]
    st.subheader(f"{selected_name} diagnostics")
    metric_columns = st.columns(6)
    for column, metric in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        value = selected[metric]
        column.metric(metric, "N/A" if value is None else f"{value:.3f}")
    if selected["auc_reason"]:
        st.info(selected["auc_reason"])
    confusion = selected["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.imshow(confusion, cmap="Blues")
    axis.set_xticks([0, 1], ["Predicted No", "Predicted Yes"])
    axis.set_yticks([0, 1], ["Actual No", "Actual Yes"])
    axis.set_title("Confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, confusion[row][column], ha="center", va="center")
    st.pyplot(figure, clear_figure=True)
    report = pd.DataFrame(selected["classification_report"]).T
    st.dataframe(report.style.format("{:.3f}"), use_container_width=True)


def main() -> None:
    st.title("Churn Signal Room")
    st.caption("A measured evaluation workspace for telecom customer retention signals")
    st.write("The positive class is **Churn = Yes**. Results describe this evaluation file; they do not guarantee an individual customer will churn.")
    try:
        metadata, models = load_artifacts()
    except Exception as exc:
        st.error(f"Saved model artifacts are unavailable. Run `python train_models.py` first. Details: {exc}")
        st.stop()

    uploaded = st.file_uploader("Upload test CSV for evaluation only", type="csv")
    if uploaded is None:
        source = "Fixed held-out test_data.csv"
        frame = pd.read_csv(TEST_PATH)
    else:
        source = "Uploaded test CSV"
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The uploaded file could not be parsed as CSV: {exc}")
            st.stop()
    frame, error = validate_evaluation_frame(frame, metadata)
    if error:
        st.error(error)
        st.stop()
    st.info(f"Evaluation source: {source} | {len(frame):,} rows | customerID is ignored by the saved pipelines.")
    names = list(models)
    selected_name = st.selectbox("Inspect a model", names)
    results = evaluate_all(models, frame, metadata)
    show_results(results, selected_name)


if __name__ == "__main__":
    main()