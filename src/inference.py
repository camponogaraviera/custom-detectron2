"""Inference entry point for Detectron2 image, video, and webcam predictions."""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from collections import Counter
from typing import Any

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from src.register import get_data_dicts, register_dataset
from src.settings import load_project_settings
from src.video_output import finalize_video_output

import matplotlib.pyplot as plt

try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

warnings.filterwarnings("ignore")

MODEL_CONFIGS: dict[str, str] = {
    "object_detection": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "instance_segmentation": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "panoptic_segmentation": "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
    "keypoint": "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
}
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "configs", "config.cfg")


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser for inference.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Type of the detectron2 model",
        required=True,
        default="object_detection",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to the Model's weights",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Detection threshold value",
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="device type",
        required=False,
        default="cpu",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Filename to run inference on image",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Filename to run inference on video",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--cam",
        type=bool,
        help="Filename to run inference on webcam",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-sg",
        "--save_gif",
        type=str,
        help="Filepath to save a gif of video inference",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-sv",
        "--save_video",
        type=str,
        help="Filepath to save a video of video inference",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vb",
        "--verbose",
        type=bool,
        help="Wether to see predictions on-the-fly",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-sf",
        "--skip_frames",
        type=int,
        help="Number of frames to skip between detections",
        required=False,
        default=1,
    )
    parser.add_argument(
        "-fb",
        "--frame_batch",
        type=int,
        help="Number of total frames to be detected",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-rf",
        "--res_factor",
        type=float,
        help="Percentage by which the output height is reduced",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--max_video_size_mb",
        type=float,
        help="Maximum size in MB for saved videos; use 0 to disable recompression",
        required=False,
        default=10.0,
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse inference CLI arguments.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        The parsed argument namespace.
    """
    return build_parser().parse_args(argv)


def _get_resized_dimensions(
    width: int,
    height: int,
    res_factor: float | None,
) -> tuple[int, int]:
    """
    Compute output dimensions from a height reduction percentage.

    Args:
        width: Original image width.
        height: Original image height.
        res_factor: Percentage by which the output height is reduced
            while preserving aspect ratio.

    Returns:
        A tuple of (new_width, new_height) for the resized output.
    """
    reduction_percentage = 0.0 if res_factor is None else float(res_factor)
    if reduction_percentage < 0 or reduction_percentage > 100:
        raise ValueError("res_factor must be between 0 and 100.")
    if reduction_percentage == 100:
        raise ValueError(
            "res_factor=100 produces a zero-height output, which cannot be encoded."
        )

    scale = 1 - reduction_percentage / 100
    new_height = int(height * scale)
    new_width = int(width * scale)
    return new_width, new_height


class Detectron:
    """
    Run Detectron2 predictions for images, videos, and webcam streams.

    Attributes:
        class_names: Class names from the configured custom dataset.
        n_classes: Number of classes in the custom dataset.
        data_path: Dataset root directory used for custom-model comparisons.
        config_path: Resolved configuration file path.
        min_size_test: Test-time minimum image size used during training.
        max_size_test: Test-time maximum image size used during training.
        model_weights: Path to custom model weights, if available.
        cfg: Active Detectron2 configuration object.
        predictor: Configured Detectron2 predictor for inference.
    """

    def __init__(
        self,
        model_type: str,
        model_weights: str | None,
        threshold: float,
        class_names: list[str],
        n_classes: int,
        data_path: str,
        config_path: str,
        min_size_test: int,
        max_size_test: int,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the predictor wrapper.

        Args:
            model_type: Detectron2 model family identifier.
            model_weights: Path to custom model weights, if available.
            threshold: Score threshold applied during inference.
            class_names: Class names from the configured custom dataset.
            n_classes: Number of classes in the custom dataset.
            data_path: Dataset root directory used for custom-model comparisons.
            config_path: Resolved configuration file path.
            min_size_test: Test-time minimum image size used during training.
            max_size_test: Test-time maximum image size used during training.
            device: Inference device, such as `cpu` or `cuda`.

        Raises:
            ValueError: If `model_type` is not supported.
        """
        self.class_names = class_names
        self.n_classes = n_classes
        self.data_path = data_path
        self.config_path = config_path
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.model_weights = model_weights
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = device
        self._load_model(model_type)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(self.cfg)

    def _get_custom_config(self) -> None:
        """
        Apply custom-dataset settings to the active configuration.

        Returns:
            None
        """
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.n_classes
        self.cfg.DATASETS.TEST = ("test_dataset",)
        self.cfg.INPUT.MIN_SIZE_TEST = self.min_size_test
        self.cfg.INPUT.MAX_SIZE_TEST = self.max_size_test

    def _load_model(self, model_type: str) -> None:
        """
        Load the requested model configuration and weights.

        Args:
            model_type: Detectron2 model family identifier.

        Returns:
            None

        Raises:
            ValueError: If `model_type` is not supported.
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError("Invalid model_type.")

        model_config = MODEL_CONFIGS[model_type]
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))

        if self.model_weights:
            model_path = os.path.join(CURRENT_DIR, self.model_weights)
            self.cfg.MODEL.WEIGHTS = model_path
            self._get_custom_config()
        else:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

    def _create_visualizer(
        self,
        img: np.ndarray,
        predictions: dict[str, Any],
    ) -> np.ndarray:
        """
        Render model predictions on an image.

        Args:
            img: Source image in OpenCV array format.
            predictions: Detectron2 prediction output for the image.

        Returns:
            The visualized image array.
        """
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            instance_mode=ColorMode.IMAGE,
        )
        visualizer.output.scale = 1.0
        visualizer._default_font_size = 35
        visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        return visualizer.output.get_image()

    def detect_objects(
        self,
        metadata: Metadata | None = None,
        predictions: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Extract detected class names from prediction output.

        Args:
            metadata: Dataset metadata used to resolve class IDs.
            predictions: Detectron2 prediction output that contains instances.

        Returns:
            A list of detected class names in prediction order.

        Raises:
            ValueError: If `predictions` is not provided.
        """
        if predictions is None:
            raise ValueError("predictions must be provided.")

        if metadata is None:
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        instances = predictions["instances"]
        detected_objects: list[str] = []
        for index in range(len(instances)):
            class_id = instances.pred_classes[index].item()
            class_name = metadata.thing_classes[class_id]
            detected_objects.append(class_name)

        return detected_objects

    def window(self, full_screen: bool = False) -> None:
        """
        Create the OpenCV result window.

        Args:
            full_screen: Whether to expand the result window to full screen.

        Returns:
            None
        """
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 800, 600)
        if full_screen:
            cv2.setWindowProperty(
                "result",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

    def _save_gif(self, gif_path: str, images: list[np.ndarray]) -> None:
        """
        Save a sequence of frames as a GIF.

        Args:
            gif_path: Output path relative to the project root.
            images: Image frames to encode in the GIF.

        Returns:
            None

        Raises:
            OSError: If the GIF cannot be written to disk.
        """
        print("Saving gif...")
        save_dir = os.path.join(CURRENT_DIR, "..", gif_path)
        imageio.mimsave(save_dir, images, format="GIF", duration=1 / 30, loop=0)
        print(f"Gif saved at {save_dir}\n")

    def _show_image(self, image_path: str) -> None:
        """
        Run inference on a single image and display the result.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            None
        """
        print("\nRunning inference on image...")
        try:
            normalized_image_path = os.path.abspath(image_path)
            img = cv2.imread(normalized_image_path)
            if img is None:
                raise ValueError("Failed to load image.")

            if self.model_weights:
                metadata = register_dataset(config_path=self.config_path)
                test_dataset_dicts = get_data_dicts(
                    os.path.join(self.data_path, "test"),
                    self.class_names,
                )
                image_dict = next(
                    (
                        dataset_dict
                        for dataset_dict in test_dataset_dicts
                        if os.path.abspath(dataset_dict["file_name"])
                        == normalized_image_path
                    ),
                    None,
                )

                predictions = self.predictor(img)
                pred_visualizer = Visualizer(
                    img[:, :, ::-1],
                    metadata=metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE_BW,
                )
                pred_visualizer = pred_visualizer.draw_instance_predictions(
                    predictions["instances"].to("cpu")
                )
                detected_objects = self.detect_objects(
                    metadata=metadata,
                    predictions=predictions,
                )

                if image_dict is not None:
                    plt.figure(figsize=(14, 10))
                    plt.subplot(121)
                    plt.imshow(
                        cv2.cvtColor(
                            pred_visualizer.get_image()[:, :, ::-1],
                            cv2.COLOR_BGR2RGB,
                        )
                    )
                    plt.title("Predicted Image")

                    img_gt = cv2.imread(image_dict["file_name"])
                    gt_visualizer = Visualizer(
                        img_gt[:, :, ::-1],
                        metadata=metadata,
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW,
                    )
                    gt_visualizer = gt_visualizer.draw_dataset_dict(image_dict)
                    plt.subplot(122)
                    plt.imshow(
                        cv2.cvtColor(
                            gt_visualizer.get_image()[:, :, ::-1],
                            cv2.COLOR_BGR2RGB,
                        )
                    )
                    plt.title("Ground Truth Image")
                else:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(
                        cv2.cvtColor(
                            pred_visualizer.get_image()[:, :, ::-1],
                            cv2.COLOR_BGR2RGB,
                        )
                    )
                    plt.title("Predicted Image")

                plt.tight_layout()
                plt.show()
            else:
                predictions = self.predictor(img)
                image_to_show = self._create_visualizer(img, predictions)
                detected_objects = self.detect_objects(predictions=predictions)
                self.window()
                cv2.imshow("result", image_to_show[:, :, ::-1])
                while True:
                    key = cv2.waitKey(1)
                    if (
                        key == 27
                        or key == ord("q")
                        or cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1
                    ):
                        break
                cv2.destroyAllWindows()

            self.count_items(detected_objects)
        except Exception as exc:
            print("Error processing image:", exc)

    def _show_video(
        self,
        video_path: str,
        save_gif: str | None,
        save_video: str | None,
        skip_frames: int | None,
        frame_batch: int | None,
        res_factor: float | None,
        max_video_size_mb: float | None,
        verbose: bool,
    ) -> None:
        """
        Run inference on a video file.

        Args:
            video_path: Absolute path to the video file.
            save_gif: Relative output path for a generated GIF, if requested.
            save_video: Output path for a rendered video, if requested.
            skip_frames: Number of frames to skip between predictions.
            frame_batch: Maximum number of processed frames to use.
            res_factor: Percentage by which the output height is reduced while
                preserving aspect ratio.
            max_video_size_mb: Maximum size budget for saved videos in MB.
            verbose: Whether to preview processed frames while generating a GIF.

        Returns:
            None

        Raises:
            ValueError: If the video file cannot be opened.
        """
        print("\nRunning inference on video...")
        skip_frames = skip_frames or 1
        frame_count = 0
        used_frames = 0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to load video file.")

        try:
            if save_video:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0

                output_dir = os.path.dirname(save_video) or "."
                os.makedirs(output_dir, exist_ok=True)
                suffix = os.path.splitext(save_video)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    dir=output_dir,
                    delete=False,
                ) as temp_output:
                    temp_output_path = temp_output.name

                writer = None
                try:
                    while True:
                        ret, image = cap.read()
                        if not ret:
                            break

                        width, height = image.shape[1], image.shape[0]
                        new_width, new_height = _get_resized_dimensions(
                            width,
                            height,
                            res_factor,
                        )
                        if (new_width, new_height) != (width, height):
                            image = cv2.resize(image, (new_width, new_height))

                        predictions = self.predictor(image)
                        image_to_show = self._create_visualizer(image, predictions)
                        frame_to_write = cv2.cvtColor(
                            image_to_show,
                            cv2.COLOR_RGB2BGR,
                        )

                        if writer is None:
                            writer = cv2.VideoWriter(
                                temp_output_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                (frame_to_write.shape[1], frame_to_write.shape[0]),
                            )
                            if not writer.isOpened():
                                raise ValueError("Failed to create output video.")

                        writer.write(frame_to_write)
                    if writer is None:
                        raise ValueError("No frames were written to the output video.")
                except Exception:
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                    raise
                finally:
                    if writer is not None:
                        writer.release()

                finalize_video_output(
                    temp_output_path,
                    video_path,
                    save_video,
                    max_video_size_mb,
                )
            elif save_gif:
                frames: list[np.ndarray] = []
                fig, ax = plt.subplots()
                predicting = True
                print("\nPredicting frames... This may take a while...")
                while predicting:
                    ret, image = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue

                    used_frames += 1
                    width, height = image.shape[1], image.shape[0]
                    new_width, new_height = _get_resized_dimensions(
                        width,
                        height,
                        res_factor,
                    )
                    if (new_width, new_height) != (width, height):
                        image = cv2.resize(image, (new_width, new_height))

                    predictions = self.predictor(image)
                    image_to_show = self._create_visualizer(image, predictions)
                    detected_objects = self.detect_objects(predictions=predictions)
                    self.count_items(detected_objects)

                    if verbose:
                        ax.imshow(image_to_show)
                        ax.axis("off")
                        plt.pause(0.2)
                        plt.draw()

                    if frame_batch and used_frames == frame_batch:
                        predicting = False

                    frames.append(image_to_show)

                plt.close(fig)
                self._save_gif(gif_path=save_gif, images=frames)
        finally:
            cap.release()

    def _show_webcam(self) -> None:
        """
        Run inference on frames streamed from the default webcam.

        Returns:
            None
        """
        print("\nRunning inference on webcam...")
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            if not ret:
                break

            predictions = self.predictor(image)
            image = self._create_visualizer(image, predictions)
            detected_objects = self.detect_objects(predictions=predictions)
            print(detected_objects)
            self.window()
            cv2.imshow("result", image)
            key = cv2.waitKey(1) & 0xFF
            if (
                key == 27
                or key == ord("q")
                or cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1
            ):
                break

        cv2.destroyAllWindows()
        cap.release()

    def count_items(self, objects: list[str]) -> None:
        """
        Print the number of detected instances for each class.

        Args:
            objects: Detected class names.

        Returns:
            None
        """
        counts = Counter(objects)
        print("\n")
        for label, count in counts.items():
            print(f"Number of {label}s detected: {count}")

    def main(
        self,
        cam: bool,
        image_name: str | None,
        video_name: str | None,
        save_gif: str | None,
        save_video: str | None,
        skip_frames: int | None,
        frame_batch: int | None,
        res_factor: float | None,
        max_video_size_mb: float | None,
        verbose: bool,
    ) -> None:
        """
        Dispatch inference to image, video, or webcam processing.

        Args:
            cam: Whether to run webcam inference.
            image_name: Project-relative path to an input image.
            video_name: Project-relative path to an input video.
            save_gif: Relative output path for a generated GIF, if requested.
            save_video: Output path for a rendered video, if requested.
            skip_frames: Number of frames to skip between predictions.
            frame_batch: Maximum number of processed frames to use.
            res_factor: Percentage by which the output height is reduced while
                preserving aspect ratio.
            max_video_size_mb: Maximum size budget for saved videos in MB.
            verbose: Whether to preview processed frames while generating a GIF.

        Returns:
            None
        """
        base_path = PROJECT_ROOT
        if image_name:
            self._show_image(os.path.abspath(os.path.join(base_path, image_name)))
        elif video_name:
            self._show_video(
                os.path.abspath(os.path.join(base_path, video_name)),
                save_gif,
                save_video,
                skip_frames,
                frame_batch,
                res_factor,
                max_video_size_mb,
                verbose,
            )
        elif cam:
            self._show_webcam()


def main(argv: list[str] | None = None) -> None:
    """
    Run the inference CLI entry point.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        None
    """
    args = _parse_args(argv)
    settings = load_project_settings(
        CONFIG_FILE_PATH,
        project_root=PROJECT_ROOT,
        current_dir=CURRENT_DIR,
    )
    detectron = Detectron(
        model_type=args.model,
        model_weights=args.weights,
        threshold=args.threshold,
        class_names=settings.class_names,
        n_classes=settings.n_classes,
        data_path=settings.data_path,
        config_path=settings.config_path,
        min_size_test=settings.min_size_test,
        max_size_test=settings.max_size_test,
        device=args.device,
    )
    detectron.main(
        args.cam,
        args.image,
        args.video,
        args.save_gif,
        args.save_video,
        args.skip_frames,
        args.frame_batch,
        args.res_factor,
        args.max_video_size_mb,
        args.verbose,
    )


if __name__ == "__main__":
    main()
