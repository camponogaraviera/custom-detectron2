"""Training entry point for custom Detectron2 models."""

from __future__ import annotations

import ast
import configparser
import os
import sys
from pathlib import Path
from typing import Any

from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer

from src.register import register_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_CONFIGS: dict[str, str] = {
    "object_detection": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "instance_segmentation": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "panoptic_segmentation": "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
    "keypoint": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
}
CONFIG_FILE_NAME = "configs/config.cfg"


def _parse_cli_args(
    argv: list[str],
    program_name: str,
) -> tuple[str, str, str]:
    """Validate training CLI arguments.

    Args:
        argv: Command-line arguments excluding the program name.
        program_name: Name of the running program for usage output.

    Returns:
        A tuple containing the model type, device, and config filename.

    Raises:
        SystemExit: If the required model type argument is missing.
        ValueError: If `model_type` is not supported.
    """
    if len(argv) < 1:
        print(f"Usage: {program_name} <model_type> <device> <config_file>")
        raise SystemExit(1)

    model_type = argv[0]
    if model_type not in MODEL_CONFIGS:
        raise ValueError("Invalid model_type.")

    device = argv[1] if len(argv) > 1 else "cpu"
    config_file = argv[2] if len(argv) > 2 else CONFIG_FILE_NAME
    return model_type, device, config_file


def _load_training_settings(config_path: str) -> dict[str, Any]:
    """Load and validate training settings from the local configuration file.

    Args:
        config_path: Absolute path to the configuration file.

    Returns:
        A dictionary containing the parsed training and model settings.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    mask_on = config.getboolean("Model", "MASK_ON")
    assert isinstance(mask_on, bool), "Assertion Error, MASK_ON is not of type Boolean."

    backbone = config.get("Model", "BACKBONE").strip("'")
    assert isinstance(backbone, str), "Assertion Error, BACKBONE is not of type string."

    depth = config.getint("Model", "DEPTH")
    assert isinstance(depth, int), "Assertion Error, DEPTH is not of type int."

    n_classes = config.getint("Training", "n_classes")
    assert isinstance(n_classes, int), "Assertion Error, n_classes is not of type int."

    iterations = config.getint("Training", "iterations")
    assert isinstance(
        iterations, int
    ), "Assertion Error, iterations is not of type int."

    steps = ast.literal_eval(config.get("Training", "steps"))
    assert isinstance(steps, tuple), "Assertion Error, steps is not of type tuple."

    n_workers = config.getint("Training", "n_workers")
    assert isinstance(n_workers, int), "Assertion Error, n_workers is not of type int."

    batch_size = config.getint("Training", "batch_size")
    assert isinstance(
        batch_size, int
    ), "Assertion Error, batch_size is not of type int."

    learning_rate = float(config.get("Training", "learning_rate"))
    assert isinstance(
        learning_rate, float
    ), "Assertion Error, learning_rate is not of type float."

    gamma = float(config.get("Training", "gamma"))
    assert isinstance(gamma, float), "Assertion Error, gamma is not of type float."

    settings: dict[str, Any] = {
        "mask_on": mask_on,
        "backbone": backbone,
        "depth": depth,
        "n_classes": n_classes,
        "iterations": iterations,
        "steps": steps,
        "n_workers": n_workers,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
    }

    if depth == 18 or 34:
        settings["channels"] = 64

    return settings


def train(
    model_config: str,
    device: str,
    current_dir: str,
    n_classes: int = 2,
    iterations: int = 300,
    steps: tuple[int, ...] = (),
    n_workers: int = 2,
    batch_size: int = 2,
    learning_rate: float = 0.00025,
    gamma: float = 0.1,
) -> None:
    """Train a Detectron2 model with the configured dataset.

    Args:
        model_config: Detectron2 model-zoo configuration identifier.
        device: Device string used by Detectron2, such as `cpu` or `cuda`.
        current_dir: Directory containing the training script.
        n_classes: Number of classes in the custom dataset.
        iterations: Maximum number of solver iterations.
        steps: Iterations where the learning rate decays.
        n_workers: Number of data loader workers.
        batch_size: Images processed per solver batch.
        learning_rate: Base learning rate for the solver.
        gamma: Multiplicative learning-rate decay factor.

    Raises:
        OSError: If the output directory cannot be created.
    """
    print("Training...")

    cfg: CfgNode = get_cfg()
    cfg.MODEL.DEVICE = device or "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.OUTPUT_DIR = os.path.abspath(os.path.join(current_dir, "..", "artifacts"))
    cfg.DATALOADER.NUM_WORKERS = n_workers
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = steps
    cfg.SOLVER.GAMMA = gamma

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


def main(argv: list[str] | None = None) -> None:
    """Run the training CLI entry point.

    Args:
        argv: Optional command-line arguments excluding the program name.
    """
    cli_args = sys.argv[1:] if argv is None else argv
    model_type, device, config_file = _parse_cli_args(cli_args, sys.argv[0])
    model_config = MODEL_CONFIGS[model_type]

    print(f"\nUsing device: {device}")
    print(f"Using config file: {config_file}\n")

    default_config_path = os.path.join(PROJECT_ROOT, "configs", "config.cfg")
    if config_file is None or not str(config_file).strip():
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"Configuration file not found. Checked: {default_config_path}"
            )
        resolved_config_path = default_config_path
    else:
        raw_path = os.fspath(config_file)
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

    settings = _load_training_settings(resolved_config_path)
    register_dataset(config_path=resolved_config_path)
    train(
        model_config=model_config,
        device=device,
        current_dir=CURRENT_DIR,
        n_classes=settings["n_classes"],
        iterations=settings["iterations"],
        steps=settings["steps"],
        n_workers=settings["n_workers"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        gamma=settings["gamma"],
    )


if __name__ == "__main__":
    main()
