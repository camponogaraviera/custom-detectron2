"""Utilities for loading and registering the custom Detectron2 dataset."""

from __future__ import annotations

import ast
import configparser
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.structures import BoxMode

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.cfg")


def _load_register_settings(
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[list[str], str]:
    """Load the dataset registration settings from the config file.

    Args:
        config_path: Optional path to the configuration file.

    Returns:
        A tuple containing the class names and dataset root path.
    """
    if config_path is None or not str(config_path).strip():
        if not os.path.exists(DEFAULT_CONFIG_PATH):
            raise FileNotFoundError(
                f"Configuration file not found. Checked: {DEFAULT_CONFIG_PATH}"
            )
        resolved_config_path = DEFAULT_CONFIG_PATH
    else:
        raw_path = os.fspath(config_path)
        candidates = (
            [raw_path]
            if os.path.isabs(raw_path)
            else [os.path.join(PROJECT_ROOT, raw_path)]
        )
        if not os.path.isabs(raw_path) and len(Path(raw_path).parts) == 1:
            candidates.extend(
                [
                    os.path.join(PROJECT_ROOT, "configs", os.path.basename(raw_path)),
                    os.path.join(CURRENT_DIR, os.path.basename(raw_path)),
                ]
            )

        resolved_config_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved_config_path = os.path.abspath(candidate)
                break

        if resolved_config_path is None:
            checked_paths = ", ".join(candidates)
            raise FileNotFoundError(
                f"Configuration file not found. Checked: {checked_paths}"
            )

    config = configparser.ConfigParser()
    config.read(resolved_config_path)

    class_names = ast.literal_eval(config.get("Register", "class_names"))
    assert isinstance(
        class_names, list
    ), "Assertion Error, class_names is not of type list."

    dir_name = config.get("Register", "dir_name").strip("'")
    data_path = os.path.abspath(os.path.join(PROJECT_ROOT, dir_name))
    assert isinstance(
        data_path, str
    ), "Assertion Error, data_path is not of type string."

    return class_names, data_path


def get_data_dicts(
    directory: str | Path,
    classes: list[str],
) -> list[dict[str, Any]]:
    """Build Detectron2 dataset dictionaries from JSON annotations.

    Args:
        directory: Directory containing annotation files and their images.
        classes: Ordered class names used to map labels to category IDs.

    Returns:
        A list of Detectron2 dataset records with polygon segmentation data.

    Raises:
        FileNotFoundError: If an annotation file cannot be opened.
        json.JSONDecodeError: If an annotation file contains invalid JSON.
        ValueError: If an annotation label is not present in `classes`.
    """
    dataset_dicts: list[dict[str, Any]] = []
    directory_path = Path(directory)

    for json_path in directory_path.glob("*.json"):
        with json_path.open(encoding="utf-8") as file:
            img_anns = json.load(file)

        record: dict[str, Any] = {
            "file_name": str(directory_path / img_anns["imagePath"]),
            "height": 224,
            "width": 224,
        }

        annotations: list[dict[str, Any]] = []
        for annotation in img_anns["shapes"]:
            points = annotation["points"]
            px = [point[0] for point in points]
            py = [point[1] for point in points]
            polygon = [coord for point in points for coord in point]

            annotations.append(
                {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [polygon],
                    "category_id": classes.index(annotation["label"]),
                    "iscrowd": 0,
                }
            )

        record["annotations"] = annotations
        dataset_dicts.append(record)

    return dataset_dicts


def register_dataset(
    config_path: str | os.PathLike[str] | None = None,
) -> Metadata:
    """Register the configured dataset splits with Detectron2.

    Args:
        config_path: Optional path to the configuration file.

    Returns:
        The metadata object for the registered training dataset.
    """
    print("Registering dataset...")
    class_names, data_path = _load_register_settings(config_path)
    registered_datasets = set(DatasetCatalog.list())
    for label in ["train", "test"]:
        dataset_name = f"{label}_dataset"
        if dataset_name not in registered_datasets:
            DatasetCatalog.register(
                dataset_name,
                partial(get_data_dicts, os.path.join(data_path, label), class_names),
            )
        MetadataCatalog.get(dataset_name).set(thing_classes=class_names)

    metadata = MetadataCatalog.get("train_dataset")
    print("Dataset registered!")
    return metadata
