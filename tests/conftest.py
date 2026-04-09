"""Pytest configuration for the inference test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the test suite.

    Args:
        config: Active pytest configuration instance.
    """
    config.addinivalue_line("markers", "tag_image: mark test as an image test")
    config.addinivalue_line("markers", "tag_video: mark test as a video test")
    config.addinivalue_line("markers", "tag_cam: mark test as a camera test")
