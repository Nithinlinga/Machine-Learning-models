# GitHub Copilot Agent Instructions — Machine Learning Assignment 2

## Role and objective

Act as the implementation agent for the Machine Learning Assignment 2 described in `ML_Assignment_2_AIML_26Oct26.pdf`. Inspect the assignment PDF and the supplied dataset before changing files. Implement a complete, reproducible classification project, including five models, evaluation metrics, saved preprocessing/model pipelines, a customized Streamlit application, tests, `requirements.txt`, and a complete `README.md`.

Do not merely explain how to do the assignment. Inspect the repository, create or update the necessary files, run the code and tests where the environment permits, and fix implementation errors. Do not claim a command, test, deployment, or result succeeded unless it actually did.

## Required dataset

Use this exact Kaggle dataset:

- Dataset page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Expected CSV filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- The same CSV may already be present in the repository or working directory. Inspect local files first. Do not require Kaggle credentials if the supplied CSV is available.
- If the CSV is not available locally, document how to download it from the Kaggle page. Do not scrape or fabricate dataset rows.

This is a binary classification task. The target column is `Churn`; predict whether a customer churns (`Yes`/`No`). The standard dataset has 7,043 records and 21 columns, including `customerID` and `Churn`. Validate the actual file rather than relying solely on these expected dimensions.

The assignment requires at least 12 input features and 500 instances. Confirm and report that the dataset meets both requirements. Exclude `customerID` from model features because it is an identifier, not a meaningful predictive feature. Use `Churn` as the target. Preserve all legitimate predictive columns unless a documented, defensible data-quality reason requires otherwise.

## Mandatory models

Train and evaluate all five models on the same training/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier — use Gaussian Naive Bayes unless the feature representation makes another supported variant more appropriate; explain the choice in the README
5. Support Vector Machine (SVM)

Use scikit-learn. Keep the implementation readable and modular. Use reasonable, documented hyperparameters. If hyperparameter tuning is performed, it must use training data only and must not leak information from the test set.

## Mandatory evaluation metrics

Calculate all six metrics for every model on the held-out test set:

- Accuracy
- ROC AUC
- Precision
- Recall
- F1 score
- Matthews Correlation Coefficient (MCC)

Display the results in a comparison table with these columns, in this order:

`ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC`

Use a consistent positive class (`Churn = Yes`) for binary precision, recall, and F1. Compute AUC from predicted probabilities or decision scores, not hard class predictions when continuous scores are available. Make the target-label mapping explicit and ensure it is consistent throughout training, evaluation, and app inference. Use a documented averaging strategy if the implementation ever changes to multiclass.

Also provide, for each model:
- Confusion matrix
- Classification report
- A brief, evidence-based observation about its performance

Identify the overall winner using a declared selection rule. Prefer the highest test ROC AUC as the primary criterion, with F1 as a stated tie-breaker, unless a different justified rule is documented. Do not choose a winner by intuition or fabricate scores. Discuss trade-offs such as false positives versus false negatives and class imbalance where supported by the measured results.

## Data preparation and leakage prevention

1. Validate the CSV exists, can be parsed, and contains `Churn`.
2. Inspect shape, column names, target values, missing values, duplicates, and data types.
3. Convert `TotalCharges` to numeric safely; blank or invalid values should become missing values and be imputed within a training-fitted preprocessing pipeline.
4. Exclude `customerID` and the target from input features.
5. Separate numeric and categorical columns programmatically.
6. Build preprocessing using a scikit-learn `ColumnTransformer` and `Pipeline` so imputers, encoders, and scalers are fitted only on training data.
7. For numeric columns, use median imputation and scale where appropriate. For categorical columns, use a suitable most-frequent imputer and one-hot encoding with safe handling of unseen categories.
8. Scale numeric features for Logistic Regression, KNN, and SVM. A shared preprocessing design may be used if appropriate, but avoid unnecessary scaling requirements for Decision Tree and Gaussian Naive Bayes.
9. Split into train and test sets with a fixed documented random seed and stratification on `Churn` (for example, 80% train / 20% test).
10. Fit every model on the same training partition and evaluate on the same held-out test partition. Do not fit or retrain models on uploaded test data.
11. Keep all test-derived metrics out of model selection/tuning. If cross-validation is used, it must run only on the training partition.
12. Handle class imbalance transparently. Do not silently resample the held-out test set. If using class weights or resampling, justify it and apply it only within training.
13. Make the training process reproducible and record the seed, split ratio, target mapping, and model parameters.

## Required Streamlit application

Create `app.py` (or `streamlit_app.py`) as a customized, functional Streamlit frontend. It must run locally and be suitable for deployment to Streamlit Community Cloud.

The app must include all of the following assignment requirements:

- CSV upload control for **test data only**.
- A model-selection dropdown for the five saved classifiers.
- Evaluation metrics for the selected model.
- A confusion matrix or classification report; preferably show both.
- Results for different models on the uploaded test data must be viewable.
- Load saved models and preprocessing artifacts. Do not retrain models on uploaded data.
- The uploaded CSV must contain every required input feature and the `Churn` target column. Validate the schema and show clear, actionable errors for missing or invalid columns.
- Compute metrics from the uploaded target and model predictions. Keep the target out of the prediction input features.
- If the app also provides a default evaluation view using the fixed held-out test set, clearly label it separately from uploaded-test results.
- Display the comparison table for all five models on the same test dataset.
- For AUC, use `predict_proba` where available and `decision_function` where needed. If AUC cannot be computed (for example, the uploaded test file contains only one target class), show `N/A` with an explanation instead of crashing.
- Ensure uploaded CSV columns are reordered or selected to match the saved pipeline's expected feature schema.
- Prevent accidental use of `customerID` as a model feature; tolerate it in uploaded CSVs by ignoring it.
- Do not expose stack traces to end users. Log or display useful validation errors.
- Make the UI distinctive and tailored to telecom customer churn rather than copying a generic template. Include concise explanations, metric definitions, model comparison, and a clear note about the positive class.
- Do not make misleading claims that model predictions guarantee a customer will churn.

The uploaded CSV is evaluation data and must contain `Churn`, as required by the assignment. Do not present a prediction-only CSV without a target as if it can support evaluation metrics.

## Required repository contents

Maintain at least the following structure, adjusting module names as needed while preserving the required files:

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── train_models.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── evaluation.py
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json
├── reports/
│   ├── metrics.json
│   └── model_comparison.csv
└── tests/
    └── test_pipeline.py
```

Keep the project straightforward. If a module is not needed, do not create it merely to fill the tree. Save each full preprocessing-plus-classifier pipeline using `joblib` (or an equivalent reliable format) so inference uses the exact training transformations. Save metadata describing expected input columns, target name/mapping, random seed, split, and model settings.

### `test_data.csv`

Create a real held-out test CSV from the selected dataset using the same reproducible stratified split as training. Include all raw input columns needed by the pipeline plus the `Churn` target; `customerID` may be retained in the CSV for traceability but must not be used for training. Do not fabricate records or use training rows as test rows. If the provided CSV is the complete dataset, split it reproducibly and save the held-out partition as `test_data.csv`.

### Model and preprocessing artifacts

Save all five complete pipelines. The Streamlit app must load these artifacts and use them directly. Ensure the model files can be loaded in the deployment environment with compatible dependency versions. Avoid absolute local paths; use paths relative to the project directory.

## README requirements

Write a complete `README.md` with these sections, in this order:

1. **Problem Statement** — frame the task as predicting telecom customer churn.
2. **Dataset Description** — include the Kaggle URL, target, actual verified dimensions, feature count used, target classes, and brief data-quality/preprocessing notes.
3. **GitHub Repository Link** — include a clearly marked placeholder if the repository URL is not known. Never invent a repository URL.
4. **Models Used** — describe each of the five models and the relevant preprocessing choices.
5. **Evaluation Metrics and Model Comparison** — include the six-metric comparison table using actual computed values.
6. **Observations on Model Performance** — one evidence-based observation for each model, plus an overall winner and the documented selection rule.
7. **How to Run Locally** — environment setup, dependency installation, training command, and Streamlit command.
8. **Repository Structure** — explain important files and saved artifacts.
9. **Streamlit App Features** — explain test CSV upload, model selection, metrics, confusion matrix/report, and evaluation without retraining.
10. **Deployment on Streamlit Community Cloud** — exact deployment steps, entrypoint, requirements, and secrets guidance if relevant.
11. **Reproducibility and Data Leakage Prevention** — split, seed, pipelines, and test-data handling.
12. **Limitations** — dataset/class-imbalance caveats and any known limitations.

Use actual metrics generated by the project. Never fill the comparison table with example, estimated, or invented numbers. If execution is not possible, leave a clearly marked pending-results placeholder and explain the exact command needed to generate results. Never claim that the app is deployed or provide a fake live link. Mark GitHub and live Streamlit URLs as pending until real URLs are supplied.

## `requirements.txt`

Include every direct runtime dependency needed by training and the Streamlit app, such as `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` (only if used), and `joblib`. Pin compatible versions where practical, and verify imports and execution. Avoid listing unused packages.

## Code quality and verification

- Use functions and modules with clear responsibilities; avoid one enormous script.
- Add meaningful comments explaining preprocessing, training, metric calculations, and app behavior.
- Validate input files and handle common errors with helpful messages.
- Use portable relative paths and avoid machine-specific paths.
- Do not commit secrets, API keys, credentials, or personal data.
- Add tests for data loading/preprocessing, model artifact loading, required metrics, schema validation, and evaluation behavior when the uploaded test set contains only one class.
- Run the training script and tests where dependencies and execution time allow.
- Confirm all five model artifacts are created and loadable.
- Confirm every model reports Accuracy, AUC, Precision, Recall, F1, and MCC, with an explicit `N/A`/explanation only when a metric is mathematically undefined.
- Confirm the app can evaluate `test_data.csv` and an independently uploaded compatible CSV without retraining.
- Ensure no data leakage, no fabricated results, and no hard-coded evaluation values.
- If a command fails, diagnose and fix the issue where possible, and record any unresolved environment limitation honestly.

## Assignment-specific deployment and submission requirements

The PDF specifies that the assignment must be performed on **BITS Virtual Lab** and requires **one screenshot** proving execution there. Do not fake or generate a screenshot as if it came from BITS Virtual Lab. The human user must run the finished project in BITS Virtual Lab and capture the genuine screenshot.

The final assignment submission is **one PDF** containing, in this order:

1. GitHub repository link, where the repository includes complete source code, `requirements.txt`, `README.md`, and `test_data.csv`.
2. Live Streamlit Community Cloud app link, which must open an interactive frontend.
3. Screenshot of assignment execution on BITS Virtual Lab.
4. The complete README content, including the dataset description, repository link, all-model metrics comparison table, per-model performance observations, and overall winner.

The PDF states the deadline is **8 November 2026 at 23:59**. It says no draft submissions or extensions will be accepted. Do not represent any submission step as completed unless it has actually been completed.

Deployment steps described by the assignment:
1. Open https://streamlit.io/cloud
2. Sign in with GitHub.
3. Choose **New App**.
4. Select the repository.
5. Select the branch (usually `main`).
6. Select `app.py`.
7. Deploy and verify the public app URL in a browser.

Before finalizing, verify the submission checklist:
- GitHub repository link works (human must supply/verify the real URL).
- Streamlit app link works and opens the interactive frontend (human must deploy and verify the real URL).
- App loads without errors.
- All required features are implemented.
- README is complete and included in the final submitted PDF.
- Genuine BITS Virtual Lab screenshot is captured and included.
- All results and observations are based on actual execution.

## Academic integrity and originality

The assignment PDF explicitly warns that GitHub commit history, identical repository structure/variable names, copied Streamlit UIs, and identical dataset/model/output combinations may be checked. It states AI tools are allowed for learning support, not direct copy-paste submissions.

Therefore:
- Produce understandable, well-commented code and explain design decisions.
- Customize the UI and README to this specific telecom churn task.
- Do not copy an existing GitHub project wholesale.
- Do not fabricate experiments, results, screenshots, commits, or deployment links.
- Make sure the student reviews, understands, runs, and can explain the implementation and results before submitting.

## Expected completion report

When finished, report:
- Files created or updated.
- Actual dataset shape and number of model input features.
- Train/test split and random seed.
- A comparison table containing the six actual metrics for all five models, if execution succeeded.
- Tests/commands run and their real outcomes.
- Any remaining manual tasks: GitHub push, Streamlit Community Cloud deployment, BITS Virtual Lab execution/screenshot, and creation of the final submission PDF.
