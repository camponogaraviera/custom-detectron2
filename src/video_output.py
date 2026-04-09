"""Utilities for saving and compressing rendered inference videos."""

from __future__ import annotations

import os
import subprocess
import tempfile


def _get_media_duration(media_path: str) -> float | None:
    """
    Return media duration in seconds, or `None` when unavailable.

    Args:
        media_path: Path to a video or audio file.

    Returns:
        Duration in seconds, or `None` if the duration cannot be determined.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None

    if duration <= 0:
        return None
    return duration


def _copy_video_with_audio(
    processed_video_path: str,
    source_video_path: str,
    output_path: str,
) -> None:
    """Attach the original audio track to a processed video when possible.

    Args:
        processed_video_path: Temporary rendered video that already has the
            correct frame geometry.
        source_video_path: Original input video used as the audio source.
        output_path: Final destination path.

    Returns:
        None.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        processed_video_path,
        "-i",
        source_video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        if os.path.exists(output_path):
            os.remove(output_path)
        os.replace(processed_video_path, output_path)
        return

    os.remove(processed_video_path)


def _has_audio_stream(media_path: str) -> bool:
    """
    Check whether a media file contains at least one audio stream.

    Args:
        media_path: Path to a video or audio file.

    Returns:
        `True` if at least one audio stream is present, `False` otherwise.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    return bool(result.stdout.strip())


def _compress_video_to_size(
    processed_video_path: str,
    source_video_path: str,
    output_path: str,
    max_video_size_mb: float,
) -> None:
    """
    Encode a processed video to fit within a target size budget.
    The frame geometry is preserved and only codec quality/bitrate
    is adjusted to reduce the final file size.

    Args:
        processed_video_path: Temporary rendered video that already
            has the correct frame geometry.
        source_video_path: Original input video used as the audio source.
        output_path: Final destination path.
        max_video_size_mb: Target maximum file size in megabytes.

    Returns:
        None.
    """
    duration = _get_media_duration(processed_video_path)
    if duration is None:
        _copy_video_with_audio(
            processed_video_path,
            source_video_path,
            output_path,
        )
        return

    target_bytes = int(max_video_size_mb * 1_000_000)
    if target_bytes <= 0:
        _copy_video_with_audio(
            processed_video_path,
            source_video_path,
            output_path,
        )
        return

    total_bitrate = int(target_bytes * 8 / duration * 0.93)
    has_audio = _has_audio_stream(source_video_path)
    audio_bitrate = 0
    if has_audio:
        audio_bitrate = min(96_000, max(48_000, total_bitrate // 8))
    video_bitrate = max(150_000, total_bitrate - audio_bitrate - 32_000)
    video_kbps = max(150, video_bitrate // 1000)
    audio_kbps = max(48, audio_bitrate // 1000) if has_audio else 0

    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_path) or ".",
        delete=False,
    ) as passlog_file:
        passlog_prefix = passlog_file.name
    os.remove(passlog_prefix)

    pass_one_command = [
        "ffmpeg",
        "-y",
        "-i",
        processed_video_path,
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{video_kbps}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        "-pass",
        "1",
        "-passlogfile",
        passlog_prefix,
        "-an",
        "-f",
        "mp4",
        os.devnull,
    ]

    pass_two_command = [
        "ffmpeg",
        "-y",
        "-i",
        processed_video_path,
    ]

    if has_audio:
        pass_two_command.extend(
            [
                "-i",
                source_video_path,
            ]
        )

    pass_two_command.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-b:v",
            f"{video_kbps}k",
            "-maxrate",
            f"{video_kbps}k",
            "-bufsize",
            f"{video_kbps * 2}k",
            "-pass",
            "2",
            "-passlogfile",
            passlog_prefix,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )

    if has_audio:
        pass_two_command.extend(
            [
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-b:a",
                f"{audio_kbps}k",
                "-shortest",
            ]
        )

    pass_two_command.append(output_path)

    try:
        subprocess.run(
            pass_one_command,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            pass_two_command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        if os.path.exists(output_path):
            os.remove(output_path)
        _copy_video_with_audio(
            processed_video_path,
            source_video_path,
            output_path,
        )
        return
    finally:
        for suffix in ("", ".log", ".log.mbtree"):
            passlog_path = f"{passlog_prefix}{suffix}"
            if os.path.exists(passlog_path):
                os.remove(passlog_path)

    if os.path.exists(processed_video_path):
        os.remove(processed_video_path)


def finalize_video_output(
    processed_video_path: str,
    source_video_path: str,
    output_path: str,
    max_video_size_mb: float | None,
) -> None:
    """
    Write the final video file with optional size-targeted compression.

    Args:
        processed_video_path: Temporary rendered video that already
            has the correct frame geometry.
        source_video_path: Original input video used as the audio source.
        output_path: Final destination path.
        max_video_size_mb: Optional target maximum file size in megabytes.
            If `None` or non-positive, no size-based compression will be applied
            and the video will simply be copied with audio

    Returns:
        None.
    """
    if max_video_size_mb is None or max_video_size_mb <= 0:
        _copy_video_with_audio(
            processed_video_path,
            source_video_path,
            output_path,
        )
        return

    _compress_video_to_size(
        processed_video_path,
        source_video_path,
        output_path,
        max_video_size_mb,
    )
