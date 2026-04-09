"""Training entry point for custom Detectron2 models."""

from __future__ import annotations

import os
import sys

import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer, hooks

from src.register import register_dataset
from src.settings import load_project_settings

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


def _resolve_num_workers(device: str, requested_workers: int) -> int:
    """
    Keep CPU training on the main process to avoid worker stalls.

    Args:
        device: Device string used by Detectron2, such as `cpu` or `cuda`.
        requested_workers: Number of data loader workers requested by the user.

    Returns:
        The number of data loader workers to use for training.
    """
    normalized_device = (device or "cpu").lower()
    if normalized_device == "cpu" and requested_workers > 0:
        print("Using 0 data loader workers for CPU training.")
        return 0
    return requested_workers


def _configure_cpu_threads(device: str, cpu_threads: int) -> None:
    """
    Configure PyTorch CPU threading for performance or reproducibility.

    Args:
        device: Device string used by Detectron2, such as `cpu` or `cuda`.
        cpu_threads: Number of PyTorch CPU threads to use for local training.

    Returns:
        None
    """
    normalized_device = (device or "cpu").lower()
    if normalized_device != "cpu" or cpu_threads <= 0:
        return
    torch.set_num_threads(cpu_threads)
    print(f"Using {cpu_threads} PyTorch CPU thread(s).")


class LocalTrainer(DefaultTrainer):
    """
    Customizes Detectron2's training loop by adjusting logging frequency.
    """

    def __init__(self, cfg: CfgNode, log_period: int = 20) -> None:
        """
        Initialize the LocalTrainer.

        Args:
            cfg: Detectron2 configuration node.
            log_period: Iteration interval used for progress logging.
        """
        self._log_period = max(1, log_period)
        super().__init__(cfg)

    def build_hooks(self) -> list[hooks.HookBase]:
        trainer_hooks = super().build_hooks()
        for index, trainer_hook in enumerate(trainer_hooks):
            if isinstance(trainer_hook, hooks.PeriodicWriter):
                trainer_hooks[index] = hooks.PeriodicWriter(
                    self.build_writers(),
                    period=self._log_period,
                )
                break
        return trainer_hooks


def build_training_cfg(
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
    min_size_train: tuple[int, ...] = (224,),
    min_size_test: int = 224,
    max_size_train: int = 224,
    max_size_test: int = 224,
) -> CfgNode:
    """Build the Detectron2 training config used by the trainer."""
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.DEVICE = device or "cpu"
    cfg.OUTPUT_DIR = os.path.abspath(os.path.join(current_dir, "..", "artifacts"))
    cfg.DATALOADER.NUM_WORKERS = _resolve_num_workers(cfg.MODEL.DEVICE, n_workers)
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
    cfg.INPUT.MIN_SIZE_TRAIN = min_size_train
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TRAIN = max_size_train
    cfg.INPUT.MAX_SIZE_TEST = max_size_test
    return cfg


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
    min_size_train: tuple[int, ...] = (224,),
    min_size_test: int = 224,
    max_size_train: int = 224,
    max_size_test: int = 224,
    cpu_threads: int = 1,
    log_period: int = 1,
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
        min_size_train: Short-edge sizes used for training augmentation.
        min_size_test: Short-edge size used during evaluation/inference.
        max_size_train: Maximum long-edge size used for training augmentation.
        max_size_test: Maximum long-edge size used during evaluation/inference.
        cpu_threads: Number of PyTorch CPU threads to use for local training.
        log_period: Iteration interval used for progress logging.

    Raises:
        OSError: If the output directory cannot be created.
    """
    print("Training...")

    _configure_cpu_threads(device, cpu_threads)

    cfg = build_training_cfg(
        model_config=model_config,
        device=device,
        current_dir=current_dir,
        n_classes=n_classes,
        iterations=iterations,
        steps=steps,
        n_workers=n_workers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        min_size_train=min_size_train,
        min_size_test=min_size_test,
        max_size_train=max_size_train,
        max_size_test=max_size_test,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = LocalTrainer(cfg, log_period=log_period)
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

    settings = load_project_settings(
        config_file,
        project_root=PROJECT_ROOT,
        current_dir=CURRENT_DIR,
        default_relative_path=CONFIG_FILE_NAME,
    )
    register_dataset(config_path=settings.config_path)
    train(
        model_config=model_config,
        device=device,
        current_dir=CURRENT_DIR,
        n_classes=settings.n_classes,
        iterations=settings.iterations,
        steps=settings.steps,
        n_workers=settings.n_workers,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        gamma=settings.gamma,
        min_size_train=settings.min_size_train,
        min_size_test=settings.min_size_test,
        max_size_train=settings.max_size_train,
        max_size_test=settings.max_size_test,
        cpu_threads=settings.cpu_threads,
        log_period=settings.log_period,
    )


if __name__ == "__main__":
    main()
