###########################################################################

'''
Cheat Sheet:

pytest 												# Starts the test runner.
pytest -v 										# Verbose.
pytest -s 										# Disable the capturing of standard output (stdout).
pytest -pdb 									# Enable the post-mortem debugger (PDB) when a test failure occurs.
pytest -m "tag_name" 					# Run all tests with a tag named "tag_name".
pytest -m "not tag_name" 			# Run all tests with exception of the ones marked with "tag_name".
pytest --junitxml=pytest_report.xml # Save test to report.

E.g.: pytest -v tests/test_inference.py
'''

import pytest
import subprocess
import shlex

@pytest.mark.tag_image
def test_image():
	command = "python src/inference.py -m 'instance_segmentation' -w '../output/model_final.pth' --image 'dataset/test/benign_2.jpg'"
	args = shlex.split(command)
	result = subprocess.run(args, capture_output=True)
	print('Test on --image.')
	assert result.returncode == 0
	
@pytest.mark.tag_video
def test_video():
	command = "python src/inference.py -m 'instance_segmentation' -t 0.8 --video 'assets/taipei.mp4' --save_gif 'assets/gif.gif' --verbose True --skip_frames 10 --frame_batch 2 --res_factor 1"
	args = shlex.split(command)
	result = subprocess.run(args, capture_output=True)
	print('Test on --video.')
	assert result.returncode == 0

@pytest.mark.tag_cam
def test_cam():
	command = "python src/inference.py -m 'panoptic_segmentation' --cam True"
	args = shlex.split(command)
	result = subprocess.run(args, capture_output=True)
	print('Test on --cam.')
	assert result.returncode == 0

if __name__ == "__main__":
	try:
		pytest.main(args=[__file__, '-s'])
		print("All tests passed successfully!")
	except AssertionError:
		print("There is room for improvement!")
