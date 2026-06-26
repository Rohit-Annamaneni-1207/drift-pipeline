import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_transaction_data(path="data/train_transaction.csv"):

    """
    Load the transaction-level fraud dataset.
    Returns:
        pd.DataFrame
    """

    df = pd.read_csv(path)
    return df

def load_identity_data(path="data/train_identity.csv"):

    """
    Load the identity-level fraud dataset.
    Returns:
        pd.DataFrame
    """

    df = pd.read_csv(path)
    return df

def merge_datasets(transaction_df, identity_df):
    """
    Merge transaction and identity tables on TransactionID.
    Args:
        transaction_df (pd.DataFrame)
        identity_df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left"
    )
    return df

def drop_high_missing_columns(df, threshold=0.95):
    """
    Drop columns with missing values above a certain threshold.
    Args:
        df (pd.DataFrame)
        threshold (float): Proportion of missing values above which to drop columns

    Returns:
        pd.DataFrame
    """
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index
    cleaned_df = df.drop(columns=cols_to_drop)
    return cleaned_df, cols_to_drop

def encode_categoricals(df):
    """
    Encode categorical columns using Label Encoding, return dataframe and maps for label encoders.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame, dict
    """

    df = df.copy()
    label_encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:

        df[col] = df[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def impute_missing_values(df):
    """
    Impute missing values in the dataframe.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numerical_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    return df

def preprocess_features(df):
    """
    Preprocess the features in the dataframe.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame, dict
    """

    df = df.copy()

    # Drop high missing columns
    df, dropped_cols = drop_high_missing_columns(df)

    # Encode categorical columns
    df, label_encoders = encode_categoricals(df)

    # Impute missing values
    df = impute_missing_values(df)

    y = df["isFraud"]

    X = df.drop(columns=["isFraud", "TransactionID"])

    return X, y, dropped_cols, label_encoders

def temporal_split(X, y, train_ratio=0.7):
    """
    Temporal split preserving transaction order.
    """

    # Sort by time explicitly
    sorted_idx = X["TransactionDT"].argsort()

    X = X.iloc[sorted_idx].reset_index(drop=True)
    y = y.iloc[sorted_idx].reset_index(drop=True)

    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_stream = X.iloc[split_idx:]
    y_stream = y.iloc[split_idx:]

    return X_train, y_train, X_stream, y_stream

def get_candidate_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }

def cross_validate_models(X_train, y_train, models, n_splits = 5):
    results = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "neg_log_loss": "neg_log_loss",
        "f1": "f1"
    }

    for name, model in models.items():
        
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        results.append({
            "model": name,
            "mean_pr_auc": np.mean(cv_results["test_pr_auc"]),
            "std_pr_auc": np.std(cv_results["test_pr_auc"]),
            "mean_roc_auc": np.mean(cv_results["test_roc_auc"]),
            "mean_logloss": -np.mean(cv_results["test_neg_log_loss"]),
            "mean_f1": np.mean(cv_results["test_f1"])
        })
    
    return pd.DataFrame(results)

def select_champion(results_df):
    results_df = results_df.sort_values(
        by=[
            "mean_pr_auc",      # primary
            "mean_roc_auc",     # tie-break 1
            "mean_logloss"      # tie-break 2
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    champion_name = results_df.iloc[0]["model"]

    return champion_name


def fit_champion(X_train, y_train, champion_name, models):
    model = models[champion_name]
    model.fit(X_train, y_train)

    return model

def save_model(model, version=1):
    path = f"outputs/model_v{version}.joblib"
    joblib.dump(model, path)

    return path

def quick_eda(df):

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    # Shape
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Fraud rate
    fraud_rate = df["isFraud"].mean()
    print(f"\nFraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

    # Missingness
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    print("\nTop 10 columns by missing %:")
    print(missing_pct.head(10))

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\nCategorical columns ({len(cat_cols)}):")
    print(cat_cols)

    # Transaction amount stats
    print("\nTransactionAmt stats:")
    print(df["TransactionAmt"].describe())

    # TransactionDT span
    print("\nTransactionDT span:")
    print(f"Min: {df['TransactionDT'].min()}")
    print(f"Max: {df['TransactionDT'].max()}")

    # Fraud over time
    plt.figure(figsize=(12, 5))
    plt.plot(df["TransactionDT"], df["isFraud"], alpha=0.2)
    plt.title("Fraud Occurrence Over Time")
    plt.xlabel("TransactionDT")
    plt.ylabel("isFraud")
    plt.show()

    # Transaction amount over time
    plt.figure(figsize=(12, 5))
    plt.scatter(
        df["TransactionDT"],
        df["TransactionAmt"],
        alpha=0.2,
        s=5
    )

    plt.title("Transaction Amount Over Time")
    plt.xlabel("TransactionDT")
    plt.ylabel("TransactionAmt")
    plt.show()

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    transaction_df = load_transaction_data()
    identity_df = load_identity_data()

    df = merge_datasets(transaction_df, identity_df)
    # quick_eda(df)
    X, y, dropped_cols, encoders = preprocess_features(df)

    print("\nFinal processed shape:", X.shape)
    print("Dropped columns:", len(dropped_cols))
    print("Remaining missing values:", X.isnull().sum().sum())

    remaining_cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("Remaining categorical columns:", len(remaining_cat_cols))

    X_train, y_train, X_stream, y_stream = temporal_split(X, y)

    print("\nTrain shape:", X_train.shape)
    print("Stream shape:", X_stream.shape)

    print("\nTrain fraud rate:", y_train.mean())
    print("Stream fraud rate:", y_stream.mean())

    print("\nTrain TransactionDT range:")
    print(X_train["TransactionDT"].min(), "→", X_train["TransactionDT"].max())

    print("\nStream TransactionDT range:")
    print(X_stream["TransactionDT"].min(), "→", X_stream["TransactionDT"].max())

    models = get_candidate_models()

    results_df = cross_validate_models(
        X_train,
        y_train,
        models
    )

    print("\nModel comparison:")
    print(results_df)

    champion_name = select_champion(results_df)

    print("\nChampion:", champion_name)

    champion_model = fit_champion(
        X_train,
        y_train,
        champion_name,
        models
    )

    model_path = save_model(champion_model)

    print("\nSaved to:", model_path)
