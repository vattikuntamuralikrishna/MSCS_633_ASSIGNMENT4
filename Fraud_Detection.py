"""
Credit Card Fraud Detection using PyOD AutoEncoder.

"""
#importing required libraries
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path('/Users/vattikunta/Downloads/creditcard2.csv')

def resolve_dataset_file(data_path: Path) -> Path:
    """Resolving a custom path to a concrete CSV file path."""
    # Fail early if the target path does not exist on disk
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    # Return immediately when the caller already passed a specific file path
    if data_path.is_file():
        return data_path

    if data_path.is_dir():
        # Prefer the standard Kaggle filename if present in the folder
        preferred_file = data_path / "creditcard.csv"
        if preferred_file.is_file():
            return preferred_file

        # Sort glob results so selection order is deterministic across runs
        csv_files = sorted(data_path.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]

        # Avoid guessing when multiple CSV files exist in the same directory
        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple CSV files found in {data_path}: {[p.name for p in csv_files]}. "
                "Pass the exact file path via --data-path."
            )

    # Raised if the path is a directory with zero CSV files or another non-file entry
    raise FileNotFoundError(f"No CSV file found at: {data_path}")

#Loads the transaction CSV, performs a stratified train/test split, and filters the training set to normal transactions (Class == 0).
#Fits a RobustScaler on the normal training features only to prevent data leakage, then transforms both normal training and full test features.

def load_and_preprocess(
    data_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """Load transactions, scale features fit on normal train data, and split."""
    csv_file = resolve_dataset_file(data_path)
    logger.info("Reading dataset from: %s", csv_file)

    df = pd.read_csv(csv_file)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in dataset.")

    feature_cols = [col for col in df.columns if col != "Class"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["Class"].to_numpy(dtype=np.int64)

    fraud_rate = float(np.mean(y == 1))
    logger.info(
        "Loaded %d rows (%d features). Fraud count: %d (%.4f%%)",
        len(df),
        len(feature_cols),
        int(np.sum(y == 1)),
        fraud_rate * 100.0,
    )

    # Stratified split preserves class imbalance across train and test sets
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Train AutoEncoder on normal transactions so reconstruction error
    # measures deviation from legitimate spending patterns
    X_train_normal = X_train_all[y_train_all == 0]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_all, y_test, scaler

#Initializes a PyOD AutoEncoder with the given layer sizes, contamination rate, training epochs, batch size, and seed.

#Logs the configuration and fits the autoencoder on the scaled normal training features so it learns to reconstruct typical transactions.

def build_and_train_model(
    X_train: np.ndarray,
    contamination: float,
    hidden_neuron_list: list[int],
    epoch_num: int,
    batch_size: int,
    random_state: int,
) -> AutoEncoder:
    """Instantiate and fit a PyOD AutoEncoder model."""
    model = AutoEncoder(
        contamination=contamination,
        hidden_neuron_list=hidden_neuron_list,
        epoch_num=epoch_num,
        batch_size=batch_size,
        random_state=random_state,
    )
    logger.info(
        "Training PyOD AutoEncoder (hidden_neuron_list=%s, epochs=%d, batch_size=%d, contamination=%.5f)",
        hidden_neuron_list,
        epoch_num,
        batch_size,
        contamination,
    )
    model.fit(X_train)
    return model

#Computes reconstruction anomaly scores and binary fraud predictions on the test set, then calculates ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix counts.
#Logs and prints the evaluation summary and classification report, returning all metrics in a dictionary for the manifest.
def evaluate_model(
    model: AutoEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Compute anomaly scores, binary predictions, and imbalance-aware metrics."""
    anomaly_scores = model.decision_function(X_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, anomaly_scores)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, anomaly_scores)
        ),
        "f1_fraud": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "precision_fraud": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_fraud": float(
            recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "threshold": float(model.threshold_),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    logger.info(
        "ROC-AUC: %.4f | PR-AUC (AUPRC): %.4f",
        metrics["roc_auc"],
        metrics["pr_auc_average_precision"],
    )
    logger.info(
        "Fraud (Class=1) -> Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["precision_fraud"],
        metrics["recall_fraud"],
        metrics["f1_fraud"],
    )
    logger.info("Confusion Matrix [TN=%d, FP=%d, FN=%d, TP=%d]", tn, fp, fn, tp)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"], digits=4
        )
    )

    return metrics


def save_manifest(output_path: Path, payload: dict[str, Any]) -> None:
    """Write run metadata, hyperparameters, and evaluation results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved experiment manifest to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fraud detection experiment using PyOD AutoEncoder."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV file or folder containing the credit card CSV dataset.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("artifacts/experiment_manifest.json"),
        help="Path to save output manifest JSON.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset held out for testing.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.00172,
        help="Expected outlier proportion (default matches ~0.172%% fraud rate in Kaggle dataset).",
    )
    parser.add_argument(
        "--hidden-neurons",
        type=int,
        nargs="+",
        default=[32, 16],
        help="Encoder hidden layer sizes (mirrored automatically by AutoEncoder).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train_all, y_test, _ = load_and_preprocess(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_and_train_model(
        X_train=X_train,
        contamination=args.contamination,
        hidden_neuron_list=args.hidden_neurons,
        epoch_num=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    manifest = {
        "model": "pyod.models.auto_encoder.AutoEncoder",
        "dataset_path": str(args.data_path),
        "split": {
            "test_size": args.test_size,
            "train_total_rows": int(len(y_train_all)),
            "train_normal_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_fraud_rows": int(np.sum(y_test == 1)),
        },
        "hyperparameters": {
            "contamination": args.contamination,
            "hidden_neuron_list": args.hidden_neurons,
            "epoch_num": args.epochs,
            "batch_size": args.batch_size,
            "random_state": args.random_state,
        },
        "metrics": metrics,
    }
    save_manifest(args.manifest_out, manifest)


if __name__ == "__main__":
    main()