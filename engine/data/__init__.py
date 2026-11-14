"""Dataset registry and loaders."""

from __future__ import annotations

from engine.data.registry import (
    DATASET_SPECS,
    DEFAULT_DATASETS_DIR,
    DatasetSpec,
    get_dataset,
    get_dataset_spec,
    is_placeholder_dataset,
    list_datasets,
    split_dataset,
)

__all__ = [
    "DATASET_SPECS",
    "DEFAULT_DATASETS_DIR",
    "DatasetSpec",
    "get_dataset",
    "get_dataset_spec",
    "is_placeholder_dataset",
    "list_datasets",
    "split_dataset",
]
