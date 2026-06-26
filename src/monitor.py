import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss
)

from src.train import (
    load_transaction_data,
    load_identity_data,
    merge_datasets,
    preprocess_features,
    temporal_split
)


# --------------------------------------------------
# Load model
# --------------------------------------------------
def load_model(path="outputs/model_v1.joblib"):
    return joblib.load(path)


# --------------------------------------------------
# Load baseline metrics
# --------------------------------------------------
def load_baseline_metrics(path="outputs/baseline_metrics.json"):
    with open(path, "r") as f:
        baseline_metrics = json.load(f)

    return baseline_metrics


# --------------------------------------------------
# Select top-k important features
# --------------------------------------------------
def select_monitor_features(X_train, model, k=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")

        if classifier is None or not hasattr(classifier, "coef_"):
            raise ValueError("Unable to extract feature importance from model.")

        importances = np.abs(classifier.coef_).ravel()
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Unable to extract feature importance from model.")

    feature_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="importance",
        ascending=False
    )

    # Remove temporal clock
    feature_importance_df = feature_importance_df[
        feature_importance_df["feature"] != "TransactionDT"
    ]

    monitor_features = feature_importance_df.head(k)["feature"].tolist()

    print("\nMonitoring features:")
    print(monitor_features)

    return monitor_features


# --------------------------------------------------
# Stream batches
# --------------------------------------------------
def simulate_stream_batches(X_stream, y_stream, batch_size=500):
    for i in range(0, len(X_stream), batch_size):
        X_batch = X_stream.iloc[i:i+batch_size]
        y_batch = y_stream.iloc[i:i+batch_size]

        if len(X_batch) == batch_size:
            yield X_batch, y_batch


# --------------------------------------------------
# Compute batch metrics
# --------------------------------------------------
def compute_batch_metrics(model, X_batch, y_batch):
    y_proba = model.predict_proba(X_batch)[:, 1]

    metrics = {
        "pr_auc": average_precision_score(y_batch, y_proba),
        "roc_auc": roc_auc_score(y_batch, y_proba),
        "log_loss": log_loss(y_batch, y_proba)
    }

    return metrics


# --------------------------------------------------
# Performance drift
# --------------------------------------------------
def detect_performance_drift(batch_metrics, baseline_metrics):
    baseline_pr_auc = baseline_metrics["pr_auc"]
    current_pr_auc = batch_metrics["pr_auc"]

    relative_drop = (
        (baseline_pr_auc - current_pr_auc)
        / baseline_pr_auc
    )

    if relative_drop > 0.20:
        return True

    return False


# --------------------------------------------------
# Feature drift
# --------------------------------------------------
def detect_feature_drift(
    prev_batch,
    current_batch,
    monitor_features,
    mean_threshold=0.35,
    std_threshold=0.35,
    min_features_drift=3
):
    drift_flags = []

    for feature in monitor_features:
        prev_mean = prev_batch[feature].mean()
        curr_mean = current_batch[feature].mean()

        prev_std = prev_batch[feature].std()
        curr_std = current_batch[feature].std()

        mean_shift = abs(curr_mean - prev_mean) / (abs(prev_mean) + 1e-8)
        std_shift = abs(curr_std - prev_std) / (abs(prev_std) + 1e-8)

        if mean_shift > mean_threshold or std_shift > std_threshold:
            drift_flags.append(feature)

    drift_detected = len(drift_flags) >= min_features_drift

    return drift_detected, drift_flags


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    # Load and preprocess
    transaction_df = load_transaction_data()
    identity_df = load_identity_data()

    df = merge_datasets(
        transaction_df,
        identity_df
    )

    X, y, _, _ = preprocess_features(df)

    X_train, y_train, X_stream, y_stream = temporal_split(X, y)

    # Load model + baseline
    model = load_model()
    baseline_metrics = load_baseline_metrics()

    # Select monitored features
    monitor_features = select_monitor_features(
        X_train,
        model
    )

    print("\nStarting stream monitoring...")
    print("=" * 50)

    retrain_triggers = []

    prev_X_batch = None
    performance_drift_streak = 0

    for batch_idx, (X_batch, y_batch) in enumerate(
        simulate_stream_batches(X_stream, y_stream)
    ):
        # Need previous batch for rolling drift comparison
        if prev_X_batch is None:
            prev_X_batch = X_batch.copy()
            continue

        # Compute batch performance
        batch_metrics = compute_batch_metrics(
            model,
            X_batch,
            y_batch
        )

        # Performance drift
        performance_drift = detect_performance_drift(
            batch_metrics,
            baseline_metrics
        )

        if performance_drift:
            performance_drift_streak += 1
        else:
            performance_drift_streak = 0

        # Feature drift
        feature_drift, drift_features = detect_feature_drift(
            prev_X_batch,
            X_batch,
            monitor_features
        )

        # Trigger only after sustained degradation
        if performance_drift_streak >= 3 and feature_drift:
            print(f"\nRetrain trigger at batch {batch_idx}")
            print("Batch metrics:", batch_metrics)
            print("Drift features:", drift_features)

            retrain_triggers.append(batch_idx)

            # Reset streak after trigger
            performance_drift_streak = 0

        # Update rolling reference
        prev_X_batch = X_batch.copy()

    print("\nTotal retrain triggers:", len(retrain_triggers))
    print("Trigger batches:", retrain_triggers)
