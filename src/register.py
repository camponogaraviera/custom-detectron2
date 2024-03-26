'''Module for registering the custom dataset in detectron2.'''

# Register dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

# get_data_dicts
from detectron2.structures import BoxMode
import json

# Config file
import configparser
import ast
import os
import numpy as np
import cv2

# Get the path to the config.cfg file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, 'config.cfg')

# Load the configuration
config = configparser.ConfigParser()
config.read(config_file_path)

# Configs for Register
class_names_str = config.get('Register', 'class_names')
class_names = ast.literal_eval(class_names_str)
assert isinstance(class_names, list), "Assertion Error, class_names is not of type list."
dir_name = config.get('Register', 'dir_name').strip("'")
data_path = os.path.join(os.path.dirname(__file__), '..', dir_name)
assert isinstance(data_path, str), "Assertion Error, data_path is not of type string."

def get_data_dicts(directory, classes):
	''' 
	Function to generate a list of data dictionaries representing annotations.
	Note: the following code snippet is sourced from https://github.com/theartificialguy/Detectron2
	'''
	dataset_dicts = []
	for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
		json_file = os.path.join(directory, filename)
		with open(json_file) as f:
			img_anns = json.load(f)

		record = {}
		
		filename = os.path.join(directory, img_anns["imagePath"])
		
		record["file_name"] = filename
		record["height"] = 224
		record["width"] = 224

		annos = img_anns["shapes"]
		objs = []
		for anno in annos:
			px = [a[0] for a in anno['points']] # x coord
			py = [a[1] for a in anno['points']] # y-coord
			poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
			poly = [p for x in poly for p in x]

			obj = {
					"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
					"bbox_mode": BoxMode.XYXY_ABS,
					"segmentation": [poly],
					"category_id": classes.index(anno['label']),
					"iscrowd": 0
			}
			objs.append(obj)
		record["annotations"] = objs
		dataset_dicts.append(record)
	return dataset_dicts

def register_dataset():
	'''
	Note: the following code snippet is adapted from Detectron2's Colab tutorial accessible via https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
  Link to the Colab: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
	'''
	print('Registering dataset...')
	for label in ['train', 'test']:
		DatasetCatalog.register(label + "_dataset", lambda label=label: get_data_dicts(data_path+label, class_names))
		MetadataCatalog.get(label + "_dataset").set(thing_classes=class_names)
	
	metadata = MetadataCatalog.get("train_dataset")
	print('Dataset registered!')
	return metadata
