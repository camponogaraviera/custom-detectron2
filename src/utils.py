"""Utility helpers for inspecting images and renaming dataset files."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_image(img_dir: str | os.PathLike[str]) -> None:
    """Display an image from disk with Matplotlib.

    Args:
        img_dir: Path to the image file to display.

    Raises:
        FileNotFoundError: If `img_dir` does not exist.
        OSError: If the image cannot be opened.
    """
    with Image.open(img_dir) as img:
        img_arr = np.array(img)

    plt.imshow(img_arr)
    plt.show()


def rename_images(
    folder_path: str | os.PathLike[str],
    extension: str = ".jpg",
) -> None:
    """Rename dataset images within the expected train and test class folders.

    The function looks for `train` and `test` subdirectories, then renames
    image files inside `benign` and `malignant` folders using sequential names
    such as `benign_1.jpg`.

    Args:
        folder_path: Root directory that contains the dataset split folders.
        extension: File extension used to filter images that should be renamed.

    Raises:
        FileNotFoundError: If a discovered directory disappears during processing.
        OSError: If a file cannot be renamed.
    """
    print("Renaming images...")
    root_path = os.fspath(folder_path)

    for subfolder in ["test", "train"]:
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.exists(subfolder_path):
            continue

        for sub_subfolder in ["benign", "malignant"]:
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
            if not os.path.exists(sub_subfolder_path):
                continue

            image_files = sorted(
                file_name
                for file_name in os.listdir(sub_subfolder_path)
                if file_name.endswith(extension)
            )

            for index, image_file in enumerate(image_files, start=1):
                file_extension = os.path.splitext(image_file)[1]
                new_file_name = f"{sub_subfolder}_{index}{file_extension}"
                old_file_path = os.path.join(sub_subfolder_path, image_file)
                new_file_path = os.path.join(sub_subfolder_path, new_file_name)
                os.rename(old_file_path, new_file_path)

    print("Done!")
