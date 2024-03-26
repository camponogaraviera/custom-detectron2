'''Utility functions'''

# Directories
import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_image(img_dir):
	# Load
	img = Image.open(img_dir)
	img_arr = np.array(img)
	# Plot
	plt.imshow(img_arr)
	plt.show()

def rename_images(folder_path, extension: str = '.jpg'):
	print('Renaming images...')
	# Get the list of subfolders (test and train)
	subfolders = ["test", "train"]
	for subfolder in subfolders:
		subfolder_path = os.path.join(folder_path, subfolder)
		# Check if the subfolder exists
		if not os.path.exists(subfolder_path):
			continue
		# Get the list of sub-subfolders (benign and malignant)
		sub_subfolders = ["benign", "malignant"]

		for sub_subfolder in sub_subfolders:
			sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
			# Check if the sub-subfolder exists
			if not os.path.exists(sub_subfolder_path):
				continue
			# Get the list of image files in the sub-subfolder
			image_files = [f for f in os.listdir(sub_subfolder_path) if f.endswith(extension)]
			# Sort the image files in alphabetical order
			image_files.sort()

			# Rename the image files in numerical order
			for i, image_file in enumerate(image_files):
				# Get the file extension
				file_extension = os.path.splitext(image_file)[1]
				# Create the new file name
				new_file_name = f"{sub_subfolder}_{i+1}{file_extension}"
				# Construct the old and new file paths
				old_file_path = os.path.join(sub_subfolder_path, image_file)
				new_file_path = os.path.join(sub_subfolder_path, new_file_name)
				# Rename the file
				os.rename(old_file_path, new_file_path)
	print('Done!')