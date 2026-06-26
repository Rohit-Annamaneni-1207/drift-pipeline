import json
import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    f1_score,
    confusion_matrix
)

from src.train import (
    load_transaction_data,
    load_identity_data,
    merge_datasets,
    preprocess_features,
    temporal_split
)

def load_model(path="outputs/model_v1.joblib"):
    model = joblib.load(path)
    return model

def evaluate_on_stream(model, X_stream, y_stream):
    y_proba = model.predict_proba(X_stream)[:, 1]
    y_pred = model.predict(X_stream)

    metrics = {
        "pr_auc": average_precision_score(y_stream, y_proba),
        "roc_auc": roc_auc_score(y_stream, y_proba),
        "log_loss": log_loss(y_stream, y_proba),
        "f1": f1_score(y_stream, y_pred),
        "confusion_matrix": confusion_matrix(
            y_stream, 
            y_pred).tolist()
    }

    return metrics

def save_baseline_metrics(metrics, path="outputs/baseline_metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    return path

if __name__ == "__main__":
    # model = load_model()

    transaction_df = load_transaction_data()
    identity_df = load_identity_data()

    df = merge_datasets(
        transaction_df, 
        identity_df
    )

    X, y, _, _ = preprocess_features(df)

    _, _, X_stream, y_stream = temporal_split(X, y)

    # load champion model
    model = load_model()

    metrics = evaluate_on_stream(
        model,
        X_stream,
        y_stream
    )

    print("Baseline Metrics:")
    print("="*40)

    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    # save baseline metrics
    save_path = save_baseline_metrics(metrics)
    print(f"\nBaseline metrics saved to {save_path}")
