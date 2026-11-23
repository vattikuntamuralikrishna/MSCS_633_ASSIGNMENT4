"""
Credit Card Fraud Detection using PyOD AutoEncoder.

Expects `creditcard.csv` from Kaggle's Credit Card Fraud Detection dataset
(columns: Time, V1..V28, Amount, Class where 1 = fraud, 0 = normal).
"""

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


def load_and_preprocess(
    data_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """Load transactions, scale features fit on normal train data, and split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
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

    # Train AutoEncoder predominantly/exclusively on normal transactions
    # so reconstruction error reflects deviation from normal behavior
    X_train_normal = X_train_all[y_train_all == 0]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_all, y_test, scaler


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


def evaluate_model(
    model: AutoEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Compute anomaly scores, binary predictions, and imbalance-aware metrics."""
    # Higher decision_scores / reconstruction error indicates stronger anomaly
    anomaly_scores = model.decision_function(X_test)
    y_pred = model.predict(X_test)

    cm = confusion