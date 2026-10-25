import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

def evaluate(model, X_test, y_test, output_dir, positive_label=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, list(model.classes_).index(positive_label)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=positive_label)),
        "recall": float(recall_score(y_test, predictions, pos_label=positive_label)),
        "f1_score": float(f1_score(y_test, predictions, pos_label=positive_label)),
        "roc_auc": float(roc_auc_score((y_test == positive_label).astype(int), scores))
    }
    (output_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir/"classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions),
                                  display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir/"confusion_matrix.png", dpi=140)
    plt.close()
    print(json.dumps(metrics, indent=2))
    return metrics
