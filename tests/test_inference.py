"""CLI regression tests for the inference entry point."""

import shlex
import subprocess
import pytest


def _run_command(command: str) -> subprocess.CompletedProcess[bytes]:
    """Execute an inference command and capture its result.

    Args:
        command: Shell-style command string passed to `shlex.split`.

    Returns:
        The completed subprocess result with captured stdout and stderr.
    """
    args = shlex.split(command)
    return subprocess.run(args, capture_output=True, check=False)


@pytest.mark.tag_image
def test_image() -> None:
    """Verify that image inference exits successfully."""
    command = (
        "python -m src.inference -m 'instance_segmentation' "
        "-w '../artifacts/model_final.pth' "
        "--image 'dataset/test/benign_2.jpg' "
    )
    result = _run_command(command)
    print("Test on --image.")
    assert result.returncode == 0


@pytest.mark.tag_video
def test_video() -> None:
    """Verify that video inference exits successfully."""
    command = (
        "python -m src.inference -m 'instance_segmentation' -t 0.8 "
        "--video 'assets/taipei.mp4' "
        "--save_gif 'assets/gif.gif' "
        "--verbose True "
        "--skip_frames 0 "
        "--frame_batch 2 "
        "--res_factor 0"
    )
    result = _run_command(command)
    print("Test on --video.")
    assert result.returncode == 0


@pytest.mark.tag_cam
def test_cam() -> None:
    """Verify that webcam inference exits successfully."""
    command = "python -m src.inference -m 'panoptic_segmentation' --cam True"
    result = _run_command(command)
    print("Test on --cam.")
    assert result.returncode == 0


if __name__ == "__main__":
    try:
        pytest.main(args=[__file__, "-s"])
        print("All tests passed successfully!")
    except AssertionError:
        print("There is room for improvement!")
