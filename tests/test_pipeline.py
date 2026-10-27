import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app import validate_evaluation_frame
from src.data_preprocessing import TARGET, feature_columns, load_dataset, schema_metadata
from src.evaluation import score_model

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_contract_and_feature_exclusion():
    frame = load_dataset(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert frame.shape == (7043, 21)
    assert len(feature_columns(frame)) == 19
    assert "customerID" not in feature_columns(frame)
    assert set(frame[TARGET]) == {"No", "Yes"}
    assert schema_metadata(frame)["target_mapping"] == {"No": 0, "Yes": 1}


def test_artifacts_and_required_metrics_exist():
    metadata_path = ROOT / "model" / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("Run train_models.py before artifact tests.")
    metadata = json.loads(metadata_path.read_text())
    for filename in metadata["artifact_files"].values():
        artifact = joblib.load(ROOT / "model" / filename)
        assert hasattr(artifact, "predict")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert len(metrics) == 5
    assert all({"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"} <= set(result) for result in metrics.values())


class OneClassModel:
    def predict(self, features):
        return [0] * len(features)

    def predict_proba(self, features):
        return [[1.0, 0.0] for _ in range(len(features))]


def test_auc_is_explicitly_undefined_for_one_class():
    result = score_model(OneClassModel(), pd.DataFrame({"value": [1, 2, 3]}), pd.Series([0, 0, 0]))
    assert result["AUC"] is None
    assert "one target class" in result["auc_reason"]


def test_uploaded_schema_requires_all_model_inputs():
    metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
    frame = pd.DataFrame({TARGET: ["No"]})
    prepared, error = validate_evaluation_frame(frame, metadata)
    assert prepared is None
    assert "Missing required columns" in error