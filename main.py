"""Educational credit scoring classification using SYNTHETIC financial data."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from evaluation import evaluate

BASE = Path(__file__).resolve().parent
FEATURES = ["annual_income", "debt_to_income", "late_payments",
            "credit_utilization", "credit_history_years"]

def create_synthetic_data(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(600000), .55, n).clip(120000, 3500000)
    debt = rng.beta(2.2, 4, n).clip(.01, .95)
    late = rng.poisson(1.3, n).clip(0, 12)
    util = rng.beta(2.5, 2.8, n)
    history = rng.uniform(.5, 22, n)
    # Invented rule, NOT real-world underwriting or validated credit labels.
    log_odds = (-1.1 + 3.0*debt + .52*late + 2.5*util
                - .09*history - .0000003*income)
    probability = 1/(1+np.exp(-log_odds))
    risky = rng.binomial(1, probability)
    return pd.DataFrame(dict(zip(FEATURES, [income, debt, late, util, history]),
                             high_credit_risk=risky))

def train():
    data_dir = BASE/"data"
    output_dir = BASE/"outputs"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    df = create_synthetic_data()
    df.to_csv(data_dir/"synthetic_credit_data.csv", index=False)
    X, y = df[FEATURES], df["high_credit_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test, output_dir)
    joblib.dump(model, output_dir/"credit_model.joblib")
    print("Saved outputs in:", output_dir)

def demo():
    model = joblib.load(BASE/"outputs"/"credit_model.joblib")
    example = pd.DataFrame([[550000, .27, 0, .23, 7]], columns=FEATURES)
    prob = model.predict_proba(example)[0, list(model.classes_).index(1)]
    print("Example (synthetic model) high-risk probability:", round(float(prob), 3))
    print("Predicted class (1 = high risk):", int(model.predict(example)[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Predict sample after training")
    args = parser.parse_args()
    demo() if args.demo else train()
