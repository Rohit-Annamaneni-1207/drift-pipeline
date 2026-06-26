# Drift Pipeline

This project is a fraud-detection monitoring prototype built on the IEEE-CIS transaction dataset. It combines:

- offline model training with scikit-learn
- pseudo-stream simulation over holdout data
- online drift detection with River
- challenger retraining and promotion logic
- an interactive Gradio interface for experimentation

The system is designed to demonstrate an end-to-end machine learning operations workflow: train a baseline model, simulate production traffic, detect deterioration, and respond with retraining logic.

## Project Goals

The repo is built around a practical MLOps story:

1. Train a fraud classifier from historical data.
2. Hold out later data to represent future traffic.
3. Feed that future data through the system in batches to mimic a live stream.
4. Measure both model quality drift and feature drift.
5. Trigger challenger retraining when degradation becomes meaningful.
6. Promote a better model when evaluation justifies it.

## Dataset

The data comes from local CSV files in [data](/Users/rohitannamaneni/projects/drift-pipeline/data):

- `train_transaction.csv`: transaction-level features and the `isFraud` target
- `train_identity.csv`: identity and device-related enrichment keyed by `TransactionID`
- `test_transaction.csv` and `test_identity.csv`: additional dataset splits included in the repo but not currently used by the app

The training pipeline merges transaction and identity records using `TransactionID`.

## Architecture

The implementation is split into focused modules in [src](/Users/rohitannamaneni/projects/drift-pipeline/src):

- [train.py](/Users/rohitannamaneni/projects/drift-pipeline/src/train.py): data loading, preprocessing, temporal split, model comparison, champion training
- [evaluate.py](/Users/rohitannamaneni/projects/drift-pipeline/src/evaluate.py): baseline evaluation metrics for the stream holdout
- [monitor.py](/Users/rohitannamaneni/projects/drift-pipeline/src/monitor.py): feature selection and drift helper functions
- [retrain.py](/Users/rohitannamaneni/projects/drift-pipeline/src/retrain.py): challenger training, evaluation, and promotion decisions
- [visualize.py](/Users/rohitannamaneni/projects/drift-pipeline/src/visualize.py): chart generation for batch performance and drift frequency
- [gradio_app.py](/Users/rohitannamaneni/projects/drift-pipeline/src/gradio_app.py): orchestration layer and interactive UI
- [__init__.py](/Users/rohitannamaneni/projects/drift-pipeline/src/__init__.py): makes `src` importable as a package

The app entrypoint is [main.py](/Users/rohitannamaneni/projects/drift-pipeline/main.py).

## Pipeline Walkthrough

### 1. Data Loading and Merge

The project starts by reading the transaction and identity CSVs and joining them on `TransactionID`. This produces a single modeling table containing labels, transaction behavior, and identity/device context.

### 2. Preprocessing

The preprocessing stage:

- drops columns above a missingness threshold
- label-encodes categorical variables
- median-imputes numeric missing values
- separates features from the `isFraud` target
- removes `TransactionID` from the feature matrix

One important design choice is that the data is split temporally using `TransactionDT`, which is more realistic than a random split for drift experiments.

### 3. Champion Model Selection

Three scikit-learn candidates are compared:

- logistic regression
- random forest
- gradient boosting

The selection metric is primarily PR-AUC, with ROC-AUC and log loss used as tie-breakers. This is appropriate for fraud detection because the dataset is imbalanced and PR-AUC is usually more informative than raw accuracy.

### 4. Stream Simulation

The later time slice is treated as future traffic and emitted in fixed-size batches. Inside the Gradio app, each batch is also iterated row by row using River’s `stream.iter_pandas`, which allows online detectors to consume a streaming signal.

### 5. Drift Detection

The current implementation uses multiple signals:

- batch-level PR-AUC deterioration relative to baseline
- River `ADWIN` on prediction errors
- River `PageHinkley` on per-row log-loss behavior
- feature-level `ADWIN` detectors over the most important model features

Retraining is triggered only when performance degradation is large enough and another drift signal agrees with it. That reduces noisy retrains.

### 6. Challenger Retraining and Promotion

When a trigger occurs, the system can:

- collect a recent retraining window from the stream
- train challenger models on that recent data
- evaluate them on a forward evaluation slice
- promote the best challenger if it beats the active champion by a required PR-AUC margin

This promotion logic is intentionally simple, which makes it easier to explain and defend in interviews.

## Gradio Application

The Gradio app lets you experiment with:

- train/stream split ratio
- number of rows used for champion training
- stream size and batch size
- number of monitored features
- PR-AUC degradation threshold
- River detector sensitivity
- auto-retraining behavior
- retraining and evaluation window sizes

The UI returns:

- a high-level run summary
- model selection results
- per-batch metrics
- trigger events
- feature drift counts
- three plots saved to [outputs](/Users/rohitannamaneni/projects/drift-pipeline/outputs)

## Outputs

The [outputs](/Users/rohitannamaneni/projects/drift-pipeline/outputs) directory stores generated artifacts such as:

- `model_v1.joblib`: serialized trained model
- `baseline_metrics.json`: baseline stream evaluation metrics
- `pr_auc_timeline.png`: PR-AUC across batches
- `retrain_triggers.png`: drift and trigger visualization
- `feature_drift_frequency.png`: feature drift frequency chart

These files are intermediate artifacts, not source code.

## Installation

The repository already contains a virtual environment, but the minimal setup flow is:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Running the App

```bash
./venv/bin/python main.py
```

This launches the Gradio interface locally.

## Dependencies

Primary Python packages:

- `scikit-learn` for training batch models
- `river` for online drift detection primitives
- `pandas` and `numpy` for data handling
- `matplotlib` for plotting
- `joblib` for model persistence
- `gradio` for the interactive application layer

See [requirements.txt](/Users/rohitannamaneni/projects/drift-pipeline/requirements.txt) for the install list.

## Design Decisions Worth Explaining

- Temporal split instead of random split because drift is inherently time-dependent.
- PR-AUC as the main metric because fraud detection is class-imbalanced.
- Batch models are kept from scikit-learn while River is used for online monitoring rather than replacing the classifier itself.
- Retraining is gated by multiple signals to avoid overreacting to noise.
- The app is intentionally parameterized so you can demonstrate sensitivity analysis live.

## Known Limitations

This project is a strong prototype, but it still has production gaps:

- categorical encoding is not persisted as a formal deployment artifact
- preprocessing is re-fit from data rather than versioned in a dedicated pipeline object
- drift thresholds are heuristic, not calibrated from historical incidents
- stream simulation is based on replayed CSV data, not a real event bus
- monitoring logic uses batches plus row-wise detectors rather than a fully online learning model
- model registry, feature store, orchestration, alerts, and audit logs are not implemented

