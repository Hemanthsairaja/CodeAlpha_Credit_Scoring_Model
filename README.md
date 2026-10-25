# CodeAlpha Task 1 — Credit Scoring Model

## Objective
Classify high credit risk (1) versus lower credit risk (0) using financial features.

## Dataset
**Synthetic, generated locally by the script**, not actual customer records.
This is a learning demonstration only; it cannot establish real creditworthiness.

## Method
Generate 2,500 synthetic examples, split into stratified 80% train and 20% test,
scale features on training data via a pipeline, fit Logistic Regression,
and evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.

## Run
```bash
pip install -r requirements.txt
python main.py
python main.py --demo
```

## Output
`data/synthetic_credit_data.csv`; `outputs/credit_model.joblib`,
`metrics.json`, `classification_report.txt`, `confusion_matrix.png`.

## Limitations
Targets are generated using an invented mathematical rule. Performance is
not evidence of fairness, suitability for lending, or real-world accuracy.
Do not use this model for actual credit decisions.
