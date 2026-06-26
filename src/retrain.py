import os
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
    temporal_split,
    get_candidate_models
)


# --------------------------------------------------
# Load current champion
# --------------------------------------------------
def load_model(path="outputs/model_v1.joblib"):
    return joblib.load(path)


# --------------------------------------------------
# Get retraining window
# --------------------------------------------------
def get_retraining_window(
    X_stream,
    y_stream,
    trigger_batch,
    batch_size=500,
    window_size=5000
):
    """
    Take last N samples before trigger.
    """

    trigger_idx = trigger_batch * batch_size

    start_idx = max(0, trigger_idx - window_size)

    X_recent = X_stream.iloc[start_idx:trigger_idx]
    y_recent = y_stream.iloc[start_idx:trigger_idx]

    return X_recent, y_recent


# --------------------------------------------------
# Train challenger models
# --------------------------------------------------
def train_challengers(X_recent, y_recent):
    models = get_candidate_models()

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining challenger: {name}")

        model.fit(X_recent, y_recent)

        trained_models[name] = model

    return trained_models


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
def evaluate_model(model, X_eval, y_eval):
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)

    metrics = {
        "pr_auc": average_precision_score(y_eval, y_proba),
        "roc_auc": roc_auc_score(y_eval, y_proba),
        "log_loss": log_loss(y_eval, y_proba)
    }

    return metrics


# --------------------------------------------------
# Evaluate challengers
# --------------------------------------------------
def evaluate_challengers(models, X_eval, y_eval):
    results = []

    for name, model in models.items():
        metrics = evaluate_model(
            model,
            X_eval,
            y_eval
        )

        results.append({
            "model": name,
            **metrics
        })

    results_df = pd.DataFrame(results)

    return results_df


# --------------------------------------------------
# Promotion logic
# --------------------------------------------------
def promote_if_better(
    champion_metrics,
    challenger_results,
    improvement_threshold=0.02
):
    """
    Promote if challenger beats champion PR-AUC
    by at least threshold.
    """

    best_challenger = challenger_results.sort_values(
        by="pr_auc",
        ascending=False
    ).iloc[0]

    if (
        best_challenger["pr_auc"]
        >
        champion_metrics["pr_auc"] + improvement_threshold
    ):
        return True, best_challenger["model"]

    return False, None


# --------------------------------------------------
# Save versioned model
# --------------------------------------------------
def save_new_champion(model, version):
    path = f"outputs/model_v{version}.joblib"

    joblib.dump(model, path)

    return path


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    # Example trigger batch
    trigger_batch = 53

    # Load full pipeline
    transaction_df = load_transaction_data()
    identity_df = load_identity_data()

    df = merge_datasets(
        transaction_df,
        identity_df
    )

    X, y, _, _ = preprocess_features(df)

    X_train, y_train, X_stream, y_stream = temporal_split(X, y)

    # Current champion
    champion_model = load_model()

    # Get recent retraining data
    X_recent, y_recent = get_retraining_window(
        X_stream,
        y_stream,
        trigger_batch
    )

    print("\nRetraining window shape:")
    print(X_recent.shape)

    # Train challengers
    challenger_models = train_challengers(
        X_recent,
        y_recent
    )

    # Use next 1000 rows as evaluation slice
    trigger_idx = trigger_batch * 500

    X_eval = X_stream.iloc[trigger_idx:trigger_idx+1000]
    y_eval = y_stream.iloc[trigger_idx:trigger_idx+1000]

    # Evaluate champion
    champion_metrics = evaluate_model(
        champion_model,
        X_eval,
        y_eval
    )

    print("\nChampion metrics:")
    print(champion_metrics)

    # Evaluate challengers
    challenger_results = evaluate_challengers(
        challenger_models,
        X_eval,
        y_eval
    )

    print("\nChallenger results:")
    print(challenger_results)

    # Promotion
    promoted, new_model_name = promote_if_better(
        champion_metrics,
        challenger_results
    )

    if promoted:
        print(f"\nPromoted new champion: {new_model_name}")

        new_model = challenger_models[new_model_name]

        save_path = save_new_champion(
            new_model,
            version=2
        )

        print("Saved:", save_path)

    else:
        print("\nChampion retained.")
