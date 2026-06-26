from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XDG_CACHE_HOME = PROJECT_ROOT / ".cache"
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
from river import drift, stream
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import evaluate_on_stream
from src.monitor import select_monitor_features, simulate_stream_batches
from src.retrain import evaluate_challengers, evaluate_model, promote_if_better, train_challengers
from src.train import (
    cross_validate_models,
    fit_champion,
    get_candidate_models,
    load_identity_data,
    load_transaction_data,
    merge_datasets,
    preprocess_features,
    select_champion,
    temporal_split,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_TRANSACTION_PATH = PROJECT_ROOT / "data" / "train_transaction.csv"
TRAIN_IDENTITY_PATH = PROJECT_ROOT / "data" / "train_identity.csv"


@lru_cache(maxsize=1)
def load_processed_training_data() -> tuple[pd.DataFrame, pd.Series]:
    transaction_df = load_transaction_data(str(TRAIN_TRANSACTION_PATH))
    identity_df = load_identity_data(str(TRAIN_IDENTITY_PATH))
    df = merge_datasets(transaction_df, identity_df)
    X, y, _, _ = preprocess_features(df)

    return X, y


def sample_training_frame(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_sample_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if training_sample_size >= len(X_train):
        return X_train, y_train

    sampled_X, _, sampled_y, _ = train_test_split(
        X_train,
        y_train,
        train_size=training_sample_size,
        stratify=y_train,
        random_state=42,
    )

    return sampled_X.reset_index(drop=True), sampled_y.reset_index(drop=True)


def train_initial_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_sample_size: int,
    cv_splits: int,
) -> tuple[object, str, pd.DataFrame]:
    sampled_X, sampled_y = sample_training_frame(X_train, y_train, training_sample_size)
    models = get_candidate_models()
    results_df = cross_validate_models(sampled_X, sampled_y, models, n_splits=cv_splits)
    champion_name = select_champion(results_df)
    champion_model = fit_champion(sampled_X, sampled_y, champion_name, models)

    return champion_model, champion_name, results_df


def safe_batch_metrics(y_true: list[int], y_proba: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in y_proba]
    unique_labels = len(set(y_true))

    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "error_rate": float(np.mean(np.not_equal(y_true, y_pred))),
        "avg_log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
    }

    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_proba))
        if unique_labels > 1
        else float("nan")
    )

    return metrics


def save_figure(fig: plt.Figure, filename: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")

    return str(path)


def build_pr_auc_figure(batch_metrics_df: pd.DataFrame, baseline_pr_auc: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(batch_metrics_df["batch_idx"], batch_metrics_df["pr_auc"], label="Batch PR-AUC")
    ax.axhline(baseline_pr_auc, linestyle="--", color="#d95f02", label="Baseline PR-AUC")

    triggers = batch_metrics_df[batch_metrics_df["retrain_trigger"]]
    if not triggers.empty:
        ax.scatter(
            triggers["batch_idx"],
            triggers["pr_auc"],
            color="#1b9e77",
            s=50,
            label="Retrain Trigger",
            zorder=3,
        )

    ax.set_title("PR-AUC Across Stream Batches")
    ax.set_xlabel("Batch")
    ax.set_ylabel("PR-AUC")
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "pr_auc_timeline.png")
    return fig


def build_drift_figure(batch_metrics_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(batch_metrics_df["batch_idx"], batch_metrics_df["error_rate"], label="Batch Error Rate")
    ax.plot(batch_metrics_df["batch_idx"], batch_metrics_df["feature_drift_count"], label="Feature Drift Count")
    ax.plot(batch_metrics_df["batch_idx"], batch_metrics_df["relative_pr_auc_drop"], label="Relative PR-AUC Drop")
    ax.set_title("River Drift Signals by Batch")
    ax.set_xlabel("Batch")
    ax.legend()
    fig.tight_layout()

    save_figure(fig, "retrain_triggers.png")
    return fig


def build_feature_figure(feature_counter: Counter) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    if feature_counter:
        features = list(feature_counter.keys())
        counts = list(feature_counter.values())
        ax.bar(features, counts, color="#7570b3")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No feature drift detected", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Feature Drift Frequency")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Drift Count")
    fig.tight_layout()

    save_figure(fig, "feature_drift_frequency.png")
    return fig


def format_summary(
    champion_name: str,
    final_model_name: str,
    baseline_metrics: dict[str, float | list[list[int]]],
    batch_metrics_df: pd.DataFrame,
    trigger_events_df: pd.DataFrame,
    promotions: int,
    training_rows_used: int,
    stream_rows_used: int,
) -> str:
    final_pr_auc = float(batch_metrics_df["pr_auc"].iloc[-1]) if not batch_metrics_df.empty else float("nan")

    return "\n".join(
        [
            "### Stream Summary",
            f"- Initial champion: `{champion_name}`",
            f"- Final active model: `{final_model_name}`",
            f"- Training rows used for selection: `{training_rows_used:,}`",
            f"- Stream rows simulated: `{stream_rows_used:,}`",
            f"- Baseline stream PR-AUC: `{baseline_metrics['pr_auc']:.4f}`",
            f"- Final batch PR-AUC: `{final_pr_auc:.4f}`",
            f"- Retrain triggers: `{len(trigger_events_df)}`",
            f"- Promotions executed: `{promotions}`",
        ]
    )


def simulate_stream_run(
    train_ratio: float,
    training_sample_size: int,
    stream_limit: int,
    batch_size: int,
    top_k_features: int,
    pr_auc_drop_threshold: float,
    min_feature_drift: int,
    auto_retrain: bool,
    retrain_window_size: int,
    eval_window_size: int,
    cv_splits: int,
    classification_threshold: float,
    adwin_delta: float,
    page_hinkley_delta: float,
    page_hinkley_threshold: float,
):
    X, y = load_processed_training_data()
    X_train, y_train, X_stream, y_stream = temporal_split(X, y, train_ratio=train_ratio)

    if stream_limit > 0:
        X_stream = X_stream.iloc[:stream_limit].reset_index(drop=True)
        y_stream = y_stream.iloc[:stream_limit].reset_index(drop=True)

    champion_model, champion_name, model_results_df = train_initial_champion(
        X_train,
        y_train,
        training_sample_size=min(training_sample_size, len(X_train)),
        cv_splits=cv_splits,
    )

    current_model = champion_model
    current_model_name = champion_name
    baseline_metrics = evaluate_on_stream(current_model, X_stream, y_stream)
    monitor_features = select_monitor_features(X_train, current_model, k=top_k_features)

    feature_detectors = {feature: drift.ADWIN(delta=adwin_delta) for feature in monitor_features}
    error_detector = drift.ADWIN(delta=adwin_delta)
    loss_detector = drift.PageHinkley(
        delta=page_hinkley_delta,
        threshold=page_hinkley_threshold,
    )

    batch_rows: list[dict[str, float | int | bool | str]] = []
    trigger_rows: list[dict[str, float | int | str]] = []
    feature_counter: Counter = Counter()
    promotions = 0

    seen_rows = 0

    for batch_idx, (X_batch, y_batch) in enumerate(
        simulate_stream_batches(X_stream, y_stream, batch_size=batch_size),
        start=1,
    ):
        y_proba = current_model.predict_proba(X_batch)[:, 1]
        batch_feature_hits: Counter = Counter()
        batch_truth = y_batch.astype(int).tolist()
        batch_scores = y_proba.tolist()
        detector_alert = False

        for (features, target), score in zip(stream.iter_pandas(X_batch, y_batch), y_proba):
            prediction = int(score >= classification_threshold)
            error_signal = float(prediction != int(target))
            clipped_score = float(np.clip(score, 1e-6, 1 - 1e-6))
            logloss_signal = float(
                -(target * np.log(clipped_score) + (1 - target) * np.log(1 - clipped_score))
            )

            error_detector.update(error_signal)
            loss_detector.update(logloss_signal)

            detector_alert = (
                detector_alert
                or error_detector.drift_detected
                or loss_detector.drift_detected
            )

            for feature in monitor_features:
                feature_detectors[feature].update(float(features[feature]))

                if feature_detectors[feature].drift_detected:
                    batch_feature_hits[feature] += 1
                    feature_counter[feature] += 1

        batch_metrics = safe_batch_metrics(batch_truth, batch_scores, classification_threshold)
        relative_drop = max(
            0.0,
            (float(baseline_metrics["pr_auc"]) - batch_metrics["pr_auc"]) / max(float(baseline_metrics["pr_auc"]), 1e-8),
        )
        feature_drift_count = len(batch_feature_hits)
        performance_drift = relative_drop >= pr_auc_drop_threshold
        retrain_trigger = performance_drift and (
            detector_alert or feature_drift_count >= min_feature_drift
        )

        batch_row = {
            "batch_idx": batch_idx,
            "model_name": current_model_name,
            "rows_seen": seen_rows + len(X_batch),
            "pr_auc": batch_metrics["pr_auc"],
            "roc_auc": batch_metrics["roc_auc"],
            "f1": batch_metrics["f1"],
            "error_rate": batch_metrics["error_rate"],
            "avg_log_loss": batch_metrics["avg_log_loss"],
            "relative_pr_auc_drop": relative_drop,
            "feature_drift_count": feature_drift_count,
            "error_drift": bool(error_detector.drift_detected),
            "loss_drift": bool(loss_detector.drift_detected),
            "retrain_trigger": retrain_trigger,
        }
        batch_rows.append(batch_row)

        if retrain_trigger:
            event = {
                "batch_idx": batch_idx,
                "rows_seen": seen_rows + len(X_batch),
                "active_model": current_model_name,
                "pr_auc": batch_metrics["pr_auc"],
                "relative_pr_auc_drop": relative_drop,
                "feature_drifts": ", ".join(sorted(batch_feature_hits)),
                "promotion": "retained",
            }

            if auto_retrain:
                trigger_end = seen_rows + len(X_batch)
                recent_start = max(0, trigger_end - retrain_window_size)
                eval_end = min(len(X_stream), trigger_end + eval_window_size)

                X_recent = X_stream.iloc[recent_start:trigger_end]
                y_recent = y_stream.iloc[recent_start:trigger_end]
                X_eval = X_stream.iloc[trigger_end:eval_end]
                y_eval = y_stream.iloc[trigger_end:eval_end]

                if len(X_recent) >= batch_size and len(X_eval) > 0 and y_eval.nunique() > 1:
                    challenger_models = train_challengers(X_recent, y_recent)
                    champion_metrics = evaluate_model(current_model, X_eval, y_eval)
                    challenger_results = evaluate_challengers(challenger_models, X_eval, y_eval)
                    promoted, new_model_name = promote_if_better(champion_metrics, challenger_results)

                    if promoted:
                        current_model = challenger_models[new_model_name]
                        current_model_name = str(new_model_name)
                        monitor_features = select_monitor_features(X_train, current_model, k=top_k_features)
                        feature_detectors = {
                            feature: drift.ADWIN(delta=adwin_delta)
                            for feature in monitor_features
                        }
                        error_detector = drift.ADWIN(delta=adwin_delta)
                        loss_detector = drift.PageHinkley(
                            delta=page_hinkley_delta,
                            threshold=page_hinkley_threshold,
                        )
                        promotions += 1
                        event["promotion"] = f"promoted:{new_model_name}"

            trigger_rows.append(event)

        seen_rows += len(X_batch)

    batch_metrics_df = pd.DataFrame(batch_rows)
    trigger_events_df = pd.DataFrame(trigger_rows)
    feature_drift_df = pd.DataFrame(
        [
            {"feature": feature, "drift_count": count}
            for feature, count in feature_counter.most_common()
        ]
    )

    summary = format_summary(
        champion_name=champion_name,
        final_model_name=current_model_name,
        baseline_metrics=baseline_metrics,
        batch_metrics_df=batch_metrics_df,
        trigger_events_df=trigger_events_df,
        promotions=promotions,
        training_rows_used=min(training_sample_size, len(X_train)),
        stream_rows_used=int(len(X_stream)),
    )

    pr_auc_fig = build_pr_auc_figure(batch_metrics_df, float(baseline_metrics["pr_auc"]))
    drift_fig = build_drift_figure(batch_metrics_df)
    feature_fig = build_feature_figure(feature_counter)

    return (
        summary,
        model_results_df,
        batch_metrics_df,
        trigger_events_df,
        feature_drift_df,
        pr_auc_fig,
        drift_fig,
        feature_fig,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Fraud Stream Monitor") as app:
        gr.Markdown(
            "\n".join(
                [
                    "# Fraud Stream Monitor",
                    "Train a batch fraud model on the IEEE-CIS data, simulate a live stream with River, and inspect drift, triggers, and challenger promotions in one place.",
                    "The app uses the local `data/train_transaction.csv` and `data/train_identity.csv` files and only loads them when you run a simulation.",
                ]
            )
        )

        with gr.Row():
            train_ratio = gr.Slider(0.5, 0.9, value=0.7, step=0.05, label="Train Ratio")
            training_sample_size = gr.Slider(10000, 150000, value=25000, step=5000, label="Training Sample Size")
            stream_limit = gr.Slider(5000, 150000, value=25000, step=5000, label="Stream Rows to Simulate")
            batch_size = gr.Slider(250, 2000, value=500, step=250, label="Batch Size")

        with gr.Row():
            top_k_features = gr.Slider(3, 20, value=10, step=1, label="Monitored Features")
            pr_auc_drop_threshold = gr.Slider(0.05, 0.5, value=0.2, step=0.01, label="Relative PR-AUC Drop Threshold")
            min_feature_drift = gr.Slider(1, 10, value=3, step=1, label="Min Feature Drifts Per Batch")
            classification_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Classification Threshold")

        with gr.Row():
            auto_retrain = gr.Checkbox(value=True, label="Auto-retrain on trigger")
            retrain_window_size = gr.Slider(1000, 20000, value=5000, step=1000, label="Retrain Window Size")
            eval_window_size = gr.Slider(500, 5000, value=1000, step=500, label="Promotion Eval Window")
            cv_splits = gr.Slider(2, 5, value=3, step=1, label="CV Splits")

        with gr.Row():
            adwin_delta = gr.Slider(0.0005, 0.02, value=0.002, step=0.0005, label="ADWIN Delta")
            page_hinkley_delta = gr.Slider(0.001, 0.05, value=0.01, step=0.001, label="PageHinkley Delta")
            page_hinkley_threshold = gr.Slider(5, 100, value=30, step=5, label="PageHinkley Threshold")

        run_button = gr.Button("Run Stream Simulation", variant="primary")

        summary = gr.Markdown()
        model_results = gr.Dataframe(label="Model Selection Results")
        batch_results = gr.Dataframe(label="Batch Metrics")
        trigger_results = gr.Dataframe(label="Trigger Events")
        feature_results = gr.Dataframe(label="Feature Drift Counts")

        with gr.Row():
            pr_auc_plot = gr.Plot(label="PR-AUC Timeline")
            drift_plot = gr.Plot(label="Drift Signals")
            feature_plot = gr.Plot(label="Feature Drift Frequency")

        run_button.click(
            fn=simulate_stream_run,
            inputs=[
                train_ratio,
                training_sample_size,
                stream_limit,
                batch_size,
                top_k_features,
                pr_auc_drop_threshold,
                min_feature_drift,
                auto_retrain,
                retrain_window_size,
                eval_window_size,
                cv_splits,
                classification_threshold,
                adwin_delta,
                page_hinkley_delta,
                page_hinkley_threshold,
            ],
            outputs=[
                summary,
                model_results,
                batch_results,
                trigger_results,
                feature_results,
                pr_auc_plot,
                drift_plot,
                feature_plot,
            ],
        )

    return app
