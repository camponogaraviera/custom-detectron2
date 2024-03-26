'''Module for training the detectron2 model on a custom dataset.'''

# Register 
from register import register_dataset

# Directories
import os

# Command-line arguments
import sys

# Training
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
	
# Config file
import configparser
import ast

if len(sys.argv) < 2:
	print(f'Usage: {sys.argv[0]} <model_type> <device> <config_file>')
	sys.exit(1)

types = {'object_detection': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml', 
				'instance_segmentation': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
				'panoptic_segmentation': 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',
				'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
				}

model_type = sys.argv[1]
if model_type in types:
	model_config = types[model_type]
else:
  raise ValueError("Invalid model_type.")
  
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
config_file = sys.argv[3] if len(sys.argv) > 3 else 'config.cfg'

print(f'\nUsing device: {device}')
print(f'Using config file: {config_file}\n')

# Get the path to the config.cfg file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, 'config.cfg')

# Load the configuration
config = configparser.ConfigParser()
config.read(config_file_path)

# Configs for Model
MASK_ON = config.getboolean('Model', 'MASK_ON')
assert isinstance(MASK_ON, bool), "Assertion Error, MASK_ON is not of type Boolean."
BACKBONE = config.get('Model', 'BACKBONE').strip("'")
assert isinstance(BACKBONE, str), "Assertion Error, BACKBONE is not of type string."
DEPTH = config.getint('Model', 'DEPTH')
assert isinstance(DEPTH, int), "Assertion Error, DEPTH is not of type int."

# Configs for Training
n_classes = config.getint('Training', 'n_classes')
assert isinstance(n_classes, int), "Assertion Error, n_classes is not of type int."
iterations = config.getint('Training', 'iterations')
assert isinstance(iterations, int), "Assertion Error, iterations is not of type int."
steps_str = config.get('Training', 'steps')
steps = ast.literal_eval(steps_str)
assert isinstance(steps, tuple), "Assertion Error, steps is not of type tuple."
n_workers = config.getint('Training', 'n_workers')
assert isinstance(n_workers, int), "Assertion Error, n_workers is not of type int."
batch_size = config.getint('Training', 'batch_size')
assert isinstance(batch_size, int), "Assertion Error, batch_size is not of type int."
learning_rate = float(config.get('Training', 'learning_rate'))
assert isinstance(learning_rate, float), "Assertion Error, learning_rate is not of type float."
gamma = float(config.get('Training', 'gamma'))
assert isinstance(gamma, float), "Assertion Error, gamma is not of type float."

if DEPTH == 18 or 34:
	CHANNELS = 64

def train(n_classes=2, iterations=300, n_workers=2, batch_size=2, learning_rate=0.00025):
	print('Training...')

	# Config
	cfg = get_cfg()
	cfg.MODEL.DEVICE = device or 'cpu'
	cfg.merge_from_file(model_zoo.get_config_file(model_config))
	cfg.DATALOADER.NUM_WORKERS = n_workers

	# Dataset
	cfg.DATASETS.TRAIN = ("train_dataset",)
	cfg.DATASETS.TEST = ()
	#cfg.DATASETS.TEST = ("test",)

	# Model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)	
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
	#cfg.MODEL.MASK_ON = MASK_ON
	#cfg.MODEL.BACKBONE.NAME = BACKBONE
	#cfg.MODEL.RESNETS.DEPTH = DEPTH
	#cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = CHANNELS # for R18/R34

	# Solver
	cfg.SOLVER.IMS_PER_BATCH = batch_size
	cfg.SOLVER.BASE_LR = learning_rate
	cfg.SOLVER.MAX_ITER = iterations
	#cfg.SOLVER.STEPS = steps or []
	#cfg.SOLVER.gamma = gamma

	# Test
	#cfg.TEST.DETECTIONS_PER_IMAGE = 5

	# Input
	#cfg.INPUT.MIN_SIZE_TRAIN = (224,)

	# Create Directory
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

	# Train
	trainer = DefaultTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()

	# Save weights to directory
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	
if __name__ == '__main__':
	# Register
	register_dataset()
	# Train
	train(n_classes, iterations, n_workers, batch_size, learning_rate)
