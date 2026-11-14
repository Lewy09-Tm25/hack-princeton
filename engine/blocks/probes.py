"""Held-out linear logistic probes with standardization across model layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Held-out evaluation metrics and unscaled linear parameters for one component."""

    component_id: str
    accuracy: float
    auroc: float
    weights: np.ndarray
    bias: float


def _to_numpy_2d(x: torch.Tensor | np.ndarray | Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {arr.shape}.")
    return arr


def _to_numpy_1d(y: torch.Tensor | np.ndarray | list[int] | Any) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        arr = y.detach().to(device="cpu").numpy()
    else:
        arr = np.asarray(y)
    return arr.reshape(-1).astype(np.int64)


def train_probe(
    X_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray | list[int],
    X_test: torch.Tensor | np.ndarray,
    y_test: torch.Tensor | np.ndarray | list[int],
    C: float = 1.0,
    component_id: str = "",
) -> ProbeResult:
    """Fit standardized L2 logistic regression on train data and score on held-out test data."""
    x_tr = _to_numpy_2d(X_train)
    y_tr = _to_numpy_1d(y_train)
    x_te = _to_numpy_2d(X_test)
    y_te = _to_numpy_1d(y_test)

    if x_tr.shape[0] != y_tr.shape[0]:
        raise ValueError("X_train and y_train must have the same number of rows.")
    if x_te.shape[0] != y_te.shape[0]:
        raise ValueError("X_test and y_test must have the same number of rows.")

    n_features = x_tr.shape[1]
    if len(np.unique(y_tr)) < 2:
        maj = int(y_tr[0]) if len(y_tr) > 0 else 0
        preds = np.full_like(y_te, fill_value=maj)
        acc = float(accuracy_score(y_te, preds)) if len(y_te) > 0 else 0.5
        return ProbeResult(
            component_id=component_id,
            accuracy=acc,
            auroc=0.5,
            weights=np.zeros(n_features, dtype=np.float32),
            bias=float(1.0 if maj == 1 else -1.0),
        )

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)
    x_te_scaled = scaler.transform(x_te)

    clf = LogisticRegression(C=float(C), max_iter=1000, random_state=0, solver="lbfgs")
    clf.fit(x_tr_scaled, y_tr)

    preds = clf.predict(x_te_scaled)
    probs = clf.predict_proba(x_te_scaled)[:, 1]
    acc = float(accuracy_score(y_te, preds))
    if len(np.unique(y_te)) >= 2:
        auroc = float(roc_auc_score(y_te, probs))
    else:
        auroc = 0.5

    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    raw_weights = (clf.coef_[0] / scale).astype(np.float32)
    raw_bias = float(clf.intercept_[0] - np.dot(raw_weights, scaler.mean_))

    return ProbeResult(
        component_id=component_id,
        accuracy=acc,
        auroc=auroc,
        weights=raw_weights,
        bias=raw_bias,
    )


def probe_all_layers(
    acts_train: dict[str, torch.Tensor | np.ndarray],
    y_train: torch.Tensor | np.ndarray | list[int],
    acts_test: dict[str, torch.Tensor | np.ndarray],
    y_test: torch.Tensor | np.ndarray | list[int],
    C: float = 1.0,
) -> list[ProbeResult]:
    """Train and evaluate one linear probe per component ID in `acts_train`."""
    results: list[ProbeResult] = []
    for comp_id, x_tr in acts_train.items():
        if comp_id not in acts_test:
            raise KeyError(f"Component {comp_id!r} present in acts_train but missing in acts_test.")
        results.append(
            train_probe(
                X_train=x_tr,
                y_train=y_train,
                X_test=acts_test[comp_id],
                y_test=y_test,
                C=C,
                component_id=comp_id,
            )
        )
    return results
