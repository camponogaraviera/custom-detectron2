###########################################################################

import pytest

def pytest_configure(config):
	config.addinivalue_line("markers", "tag_image: mark test as an image test")
	config.addinivalue_line("markers", "tag_video: mark test as a video test")
	config.addinivalue_line("markers", "tag_cam: mark test as a camera test")
