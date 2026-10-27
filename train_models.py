"""Train the five required churn classifiers and write reproducible artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import RANDOM_STATE, TARGET, TEST_SIZE, build_preprocessor, load_dataset, prepare_frame, schema_metadata, split_dataset
from src.evaluation import comparison_row, score_model

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


def model_definitions():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE), False),
        "KNN": (KNeighborsClassifier(n_neighbors=15), True),
        "Naive Bayes": (GaussianNB(), False),
        "SVM": (SVC(kernel="rbf", probability=True, C=1.0, random_state=RANDOM_STATE), True),
    }


def main() -> None:
    frame = load_dataset(DATASET_PATH)
    prepared = prepare_frame(frame)
    train_features, test_features, train_target, test_target = split_dataset(prepared)
    _, raw_test = train_test_split(prepared, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=prepared[TARGET])
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    raw_test.to_csv(ROOT / "test_data.csv", index=False)
    results = {}
    settings = {}
    file_names = {"Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib", "KNN": "knn.joblib", "Naive Bayes": "naive_bayes.joblib", "SVM": "svm.joblib"}
    for name, (classifier, scale_numeric) in model_definitions().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(prepared, scale_numeric=scale_numeric)), ("classifier", classifier)])
        pipeline.fit(train_features, train_target)
        results[name] = score_model(pipeline, test_features, test_target)
        settings[name] = str(classifier)
        joblib.dump(pipeline, MODEL_DIR / file_names[name])
    comparison = pd.DataFrame([comparison_row(name, result) for name, result in results.items()])
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    metadata = {**schema_metadata(prepared), "dataset_shape": list(frame.shape), "input_feature_count": len(schema_metadata(prepared)["input_columns"]), "train_rows": len(train_features), "test_rows": len(test_features), "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "models": settings, "artifact_files": file_names}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison.to_string(index=False))
    print(f"Dataset shape: {frame.shape}; input features: {metadata['input_feature_count']}")


if __name__ == "__main__":
    main()