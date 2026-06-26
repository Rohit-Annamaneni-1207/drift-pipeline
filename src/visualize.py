import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import average_precision_score

from src.train import (
    load_transaction_data,
    load_identity_data,
    merge_datasets,
    preprocess_features,
    temporal_split
)

from src.monitor import (
    load_baseline_metrics,
    select_monitor_features,
    simulate_stream_batches,
    detect_performance_drift,
    detect_feature_drift
)


# --------------------------------------------------
# Load model
# --------------------------------------------------
def load_model(path="outputs/model_v1.joblib"):
    return joblib.load(path)


# --------------------------------------------------
# Compute PR-AUC across all batches
# --------------------------------------------------
def collect_batch_metrics(model, X_stream, y_stream):
    batch_pr_aucs = []

    for X_batch, y_batch in simulate_stream_batches(
        X_stream,
        y_stream
    ):
        y_proba = model.predict_proba(X_batch)[:, 1]

        pr_auc = average_precision_score(
            y_batch,
            y_proba
        )

        batch_pr_aucs.append(pr_auc)

    return batch_pr_aucs


# --------------------------------------------------
# Detect retrain triggers + feature drift counts
# --------------------------------------------------
def collect_drift_events(
    model,
    X_train,
    X_stream,
    y_stream,
    baseline_metrics
):
    monitor_features = select_monitor_features(
        X_train,
        model
    )

    trigger_batches = []
    feature_counter = Counter()

    prev_X_batch = None
    performance_drift_streak = 0

    for batch_idx, (X_batch, y_batch) in enumerate(
        simulate_stream_batches(X_stream, y_stream)
    ):
        if prev_X_batch is None:
            prev_X_batch = X_batch.copy()
            continue

        y_proba = model.predict_proba(X_batch)[:, 1]
        pr_auc = average_precision_score(
            y_batch,
            y_proba
        )

        batch_metrics = {
            "pr_auc": pr_auc
        }

        performance_drift = detect_performance_drift(
            batch_metrics,
            baseline_metrics
        )

        if performance_drift:
            performance_drift_streak += 1
        else:
            performance_drift_streak = 0

        feature_drift, drift_features = detect_feature_drift(
            prev_X_batch,
            X_batch,
            monitor_features
        )

        for feature in drift_features:
            feature_counter[feature] += 1

        if performance_drift_streak >= 3 and feature_drift:
            trigger_batches.append(batch_idx)
            performance_drift_streak = 0

        prev_X_batch = X_batch.copy()

    return trigger_batches, feature_counter


# --------------------------------------------------
# Plot PR-AUC timeline
# --------------------------------------------------
def plot_pr_auc(batch_pr_aucs, baseline_pr_auc):
    plt.figure(figsize=(14, 6))

    plt.plot(batch_pr_aucs, label="Batch PR-AUC")
    plt.axhline(
        y=baseline_pr_auc,
        linestyle="--",
        label="Baseline PR-AUC"
    )

    plt.xlabel("Batch Index")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC Over Stream Batches")
    plt.legend()

    plt.savefig("outputs/pr_auc_timeline.png")
    plt.close()


# --------------------------------------------------
# Plot retrain triggers
# --------------------------------------------------
def plot_retrain_triggers(batch_pr_aucs, trigger_batches):
    plt.figure(figsize=(14, 6))

    plt.plot(batch_pr_aucs, label="Batch PR-AUC")

    for trigger in trigger_batches:
        plt.axvline(
            x=trigger,
            linestyle="--"
        )

    plt.xlabel("Batch Index")
    plt.ylabel("PR-AUC")
    plt.title("Retrain Trigger Timeline")
    plt.legend()

    plt.savefig("outputs/retrain_triggers.png")
    plt.close()


# --------------------------------------------------
# Plot feature drift frequency
# --------------------------------------------------
def plot_feature_drift(feature_counter):
    features = list(feature_counter.keys())
    counts = list(feature_counter.values())

    plt.figure(figsize=(12, 6))

    plt.bar(features, counts)

    plt.xticks(rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Drift Count")
    plt.title("Feature Drift Frequency")

    plt.savefig("outputs/feature_drift_frequency.png")
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    transaction_df = load_transaction_data()
    identity_df = load_identity_data()

    df = merge_datasets(
        transaction_df,
        identity_df
    )

    X, y, _, _ = preprocess_features(df)

    X_train, y_train, X_stream, y_stream = temporal_split(X, y)

    model = load_model()
    baseline_metrics = load_baseline_metrics()

    # Batch metrics
    batch_pr_aucs = collect_batch_metrics(
        model,
        X_stream,
        y_stream
    )

    # Drift events
    trigger_batches, feature_counter = collect_drift_events(
        model,
        X_train,
        X_stream,
        y_stream,
        baseline_metrics
    )

    # Plots
    plot_pr_auc(
        batch_pr_aucs,
        baseline_metrics["pr_auc"]
    )

    plot_retrain_triggers(
        batch_pr_aucs,
        trigger_batches
    )

    plot_feature_drift(
        feature_counter
    )

    print("\nSaved:")
    print("outputs/pr_auc_timeline.png")
    print("outputs/retrain_triggers.png")
    print("outputs/feature_drift_frequency.png")
