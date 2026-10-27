# Telecom Customer Churn Classification

## Problem Statement

This project predicts whether a telecom customer will churn (`Yes`) or remain (`No`) from account, service, contract, and billing attributes. It is an evaluation project, not a guarantee about any individual customer.

## Dataset Description

The project uses the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The supplied CSV was validated as 7,043 rows by 21 columns. `Churn` is the binary target with classes `No` and `Yes`; 19 input features are used after excluding `customerID` as an identifier. `TotalCharges` is converted to numeric, with invalid or blank values treated as missing and imputed inside each training-fitted pipeline. Categorical values are most-frequent imputed and one-hot encoded with unseen-category handling.

The dataset exceeds the assignment minimums of 500 instances and 12 input features. If the CSV is not present, download it from the Kaggle URL and place it at `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## GitHub Repository Link

**Repository URL: pending human-supplied link**

## Models Used

- **Logistic Regression:** scaled numeric features and one-hot categorical features.
- **Decision Tree Classifier:** median-imputed numeric and one-hot categorical features without numeric scaling.
- **K-Nearest Neighbors:** scaled numeric and one-hot categorical features.
- **Gaussian Naive Bayes:** dense one-hot features, with numeric imputation and no unnecessary scaling.
- **Support Vector Machine:** RBF SVM with probability estimates, scaled numeric and one-hot categorical features.

All five are complete scikit-learn pipelines. They use the same stratified split and are fitted only on the training partition.

## Evaluation Metrics and Model Comparison

The table below records the supplied held-out evaluation comparison. The positive class is `Churn = Yes`. AUC uses predicted probabilities; precision, recall, and F1 use binary averaging with `Yes` as positive. The primary selection rule for this comparison is strongest overall classification using accuracy and precision together, while AUC, F1, recall, and MCC explain the trade-offs.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.848 | 0.661 | 0.552 | 0.602 | 0.478 |
| Decision Tree | 0.805 | 0.860 | 0.669 | 0.524 | 0.588 | 0.469 |
| KNN | 0.805 | 0.857 | 0.643 | 0.601 | 0.621 | 0.491 |
| Naive Bayes | 0.696 | 0.819 | 0.460 | 0.847 | 0.596 | 0.431 |
| Support Vector Machine | 0.815 | 0.845 | 0.702 | 0.523 | 0.600 | 0.491 |

## Observations on Model Performance

- **Logistic Regression:** Provides strong overall performance with 0.806 accuracy and 0.848 AUC. Its 0.661 precision is good, although the 0.552 recall means some churners are missed.
- **Decision Tree:** Achieves the highest AUC at 0.860 and the highest precision among the tree-based and linear alternatives at 0.669. Its moderate recall of 0.524 indicates a balanced but relatively conservative churn signal.
- **KNN:** Produces a strong F1 score of 0.621 and balanced precision and recall values of 0.643 and 0.601. Its MCC of 0.491 indicates useful agreement beyond chance, although prediction cost may increase for larger datasets.
- **Naive Bayes:** Detects the most actual churners, with the highest recall of 0.847. However, its low precision of 0.460 indicates many false-positive retention alerts, which also lowers accuracy to 0.696.
- **Support Vector Machine:** Has the highest accuracy (0.815) and highest precision (0.702), so its positive predictions are comparatively reliable. Its recall of 0.523 is moderate, meaning it is less suitable when missing potential churners is the main concern.

**Overall winner:** Support Vector Machine based on the highest accuracy (0.815) and precision (0.702) in this comparison. Decision Tree has the highest ROC AUC (0.860), while KNN has the strongest F1/MCC combination. The preferred model therefore depends on the business objective: SVM for reliable overall classification, Decision Tree for ranking discrimination, KNN for balanced classification, or Naive Bayes when maximizing churn recall is the priority. Recall is important because missed churners are costly, while precision reflects the efficiency of retention outreach; the class distribution should be considered when interpreting all metrics.

## How to Run Locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Train and save all pipelines with `python train_models.py`.
4. Run tests with `pytest -q`.
5. Launch the app with `streamlit run app.py`.

## Repository Structure

- `train_models.py` trains all five models, writes `test_data.csv`, and creates artifacts and reports.
- `src/data_preprocessing.py` validates data and builds leakage-safe transformers.
- `src/evaluation.py` calculates the required metrics and diagnostics.
- `model/` contains complete joblib pipelines and metadata.
- `reports/` contains JSON diagnostics and the comparison CSV.
- `app.py` provides the test-only Streamlit evaluation interface.
- `tests/` covers dataset contracts, artifact loading, metrics, and one-class AUC behavior.

## Streamlit App Features

The app loads saved pipelines without retraining. It starts with the reproducible held-out `test_data.csv`, and also accepts a CSV upload for evaluation only. An uploaded file must include every model input column and `Churn`; `customerID` is tolerated and ignored. The interface shows the all-model comparison table, a selected model's metrics, confusion matrix, and classification report. If an evaluation file has one target class, AUC is shown as `N/A` with an explanation.

## Deployment on Streamlit Community Cloud

Push the repository, then open <https://streamlit.io/cloud>, sign in with GitHub, choose **New App**, select the repository and branch, choose `app.py`, and deploy. The committed `requirements.txt` supplies dependencies. This app requires no secrets. The public app URL is **pending human deployment and verification**.

## Reproducibility and Data Leakage Prevention

The split is stratified, 80% train and 20% test, with random seed `42`. Every model uses a pipeline containing a `ColumnTransformer`; imputers, encoders, and scalers are fitted only on training data. The held-out test rows are written to `test_data.csv` for evaluation. The Streamlit upload path only predicts and scores with saved artifacts and never fits on uploaded data.

## Limitations

This is one historical customer population and may not generalize to another telecom provider or time period. Churn is imbalanced, so accuracy alone is insufficient. The app evaluates files with known labels and is not a prediction-only service. GitHub publishing, Streamlit deployment, the genuine BITS Virtual Lab screenshot, and the final PDF remain manual submission tasks.