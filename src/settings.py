"""Shared configuration loading for training, registration, and inference."""

from __future__ import annotations

import ast
import configparser
import os
from dataclasses import dataclass

from src.utils.parse_int_tuple import parse_int_tuple
from src.utils.resolve_config_path import resolve_config_path

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.cfg")


@dataclass(frozen=True)
class ProjectSettings:
    """
    Typed settings loaded from the project configuration file.

    Provides clean attribute access to all shared settings for training,
    registration, and inference. Example: settings.n_classes

    Attributes:
        config_path: The resolved path to the loaded config file.
        class_names: A list of class names for the custom dataset.
        data_path: The absolute path to the dataset directory.
        n_classes: The number of classes in the dataset.
        iterations: The number of training iterations.
        steps: A tuple of iteration steps for learning rate scheduling.
        n_workers: The number of worker threads for data loading.
        batch_size: The training batch size.
        learning_rate: The initial learning rate for training.
        gamma: The learning rate decay factor for training.
        min_size_train: A tuple of minimum image sizes for training.
        min_size_test: The minimum image size for testing.
        max_size_train: The maximum image size for training.
        max_size_test: The maximum image size for testing.
        cpu_threads: The number of CPU threads to use for training and inference.
        log_period: The number of iterations between logging training metrics.
    """

    config_path: str
    class_names: list[str]
    data_path: str
    n_classes: int
    iterations: int
    steps: tuple[int, ...]
    n_workers: int
    batch_size: int
    learning_rate: float
    gamma: float
    min_size_train: tuple[int, ...]
    min_size_test: int
    max_size_train: int
    max_size_test: int
    cpu_threads: int
    log_period: int


def load_project_settings(
    config_path: str | os.PathLike[str] | None,
    *,
    project_root: str | os.PathLike[str],
    current_dir: str | os.PathLike[str],
    default_relative_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
) -> ProjectSettings:
    """
    Load the shared project settings from the configured config file.

    Args:
        config_path: Optional path to the config file. If None, project root,
            current directory, and default relative path are used to resolve
            the config path.
        project_root: The root directory of the project, used for resolving
            the config path and data path.
        current_dir: The current working directory, used for resolving
            the config path.
        default_relative_path: The default relative path to the config file
            from the project root.

    Returns:
        A ProjectSettings instance containing all the loaded settings.

    Raises:
        AssertionError: If any of the loaded settings do not match the expected types.
    """
    resolved_config_path = resolve_config_path(
        config_path,
        project_root=project_root,
        current_dir=current_dir,
        default_relative_path=default_relative_path,
    )
    project_root_path = os.path.abspath(os.fspath(project_root))

    config = configparser.ConfigParser()
    config.read(resolved_config_path)

    class_names = ast.literal_eval(config.get("Register", "class_names"))
    assert isinstance(
        class_names, list
    ), "Assertion Error, class_names is not of type list."

    dir_name = config.get("Register", "dir_name").strip("'")
    data_path = os.path.abspath(os.path.join(project_root_path, dir_name))
    assert isinstance(
        data_path, str
    ), "Assertion Error, data_path is not of type string."

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

    min_size_train = parse_int_tuple(
        config.get("Training", "min_size_train", fallback="(224,)")
    )
    min_size_test = config.getint("Training", "min_size_test", fallback=224)
    max_size_train = config.getint("Training", "max_size_train", fallback=224)
    max_size_test = config.getint("Training", "max_size_test", fallback=224)
    cpu_threads = config.getint("Training", "cpu_threads", fallback=1)
    log_period = config.getint("Training", "log_period", fallback=1)

    return ProjectSettings(
        config_path=resolved_config_path,
        class_names=class_names,
        data_path=data_path,
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
        cpu_threads=cpu_threads,
        log_period=log_period,
    )
