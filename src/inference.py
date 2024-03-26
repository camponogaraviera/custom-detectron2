'''Module to run inference'''

import os, argparse, cv2, imageio.v2 as imageio, warnings
from collections import Counter

# Utils
from register import get_data_dicts

# Register 
from register import register_dataset

# Inference
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Save predicted video
from detectron2.utils.video_visualizer import VideoVisualizer
from moviepy.editor import VideoFileClip

# Plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Config file
import configparser
import ast
import os 

# Ignoring all warning messages:
warnings.filterwarnings("ignore")
# Argparse:
parser = argparse.ArgumentParser(description='Run inference.')
parser.add_argument('-m',  '--model', type=str, help='Type of the detectron2 model', required=True, default='object_detection')
parser.add_argument('-w',  '--weights', type=str, help='Path to the Model\'s weights', required=False, default=None)
parser.add_argument('-t',  '--threshold', type=float, help='Detection threshold value', required=False, default=0.5)
parser.add_argument('-d',  '--device', type=str, help='device type', required=False, default='cpu')
parser.add_argument('-i',  '--image', type=str, help='Filename to run inference on image', required=False, default=None)
parser.add_argument('-v',  '--video', type=str, help='Filename to run inference on video', required=False, default=None)
parser.add_argument('-c',  '--cam', type=bool, help='Filename to run inference on webcam', required=False, default=False)
parser.add_argument('-sg', '--save_gif', type=str, help='Filepath to save a gif of video inference', required=False, default=None)
parser.add_argument('-sv', '--save_video', type=str, help='Filepath to save a video of video inference', required=False, default=None)
parser.add_argument('-vb', '--verbose', type=bool, help='Wether to see predictions on-the-fly', required=False, default=False)
parser.add_argument('-sf', '--skip_frames', type=int, help='Number of frames to skip between detections', required=False, default=1)
parser.add_argument('-fb', '--frame_batch', type=int, help='Number of total frames to be detected', required=False, default=None)
parser.add_argument('-rf', '--res_factor', type=float, help='Factor by which the video resolution is reduced', required=False, default=1)
args = parser.parse_args()

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

# Configs for inference
n_classes = config.getint('Training', 'n_classes')
assert isinstance(n_classes, int), "Assertion Error, n_classes is not of type int."
	
dir_name = config.get('Register', 'dir_name').strip("'")
data_path = os.path.join(os.path.dirname(__file__), '..', dir_name)
assert isinstance(data_path, str), "Assertion Error, data_path is not of type string."

class Detectron:
	def __init__(self, model_type, model_weights, threshold, device='cpu'):
		self.model_weights = model_weights
		self.cfg = self._get_default_config(device)
		self._load_model(model_type)
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
		self.predictor = DefaultPredictor(self.cfg)
		
	def _get_default_config(self, device):
		cfg = get_cfg()
		cfg.MODEL.DEVICE = device  # 'cuda'
		return cfg

	def _get_custon_config(self):
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes  # Number of classes in the custom dataset.
		self.cfg.DATASETS.TEST = ("skin_test", )
		  
	def _load_model(self, model_type):
		types = {'object_detection': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml', 
						'instance_segmentation': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
						'panoptic_segmentation': 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml',
						'keypoint': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
						}
		if model_type in types:
			model_config = types[model_type]
		else:
			raise ValueError("Invalid model_type.")
		# Load config file:
		self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
		# Load pretrained model:
		if self.model_weights:
			model_path = os.path.join(os.path.dirname(__file__), args.weights) 
			self.cfg.MODEL.WEIGHTS = model_path
			self._get_custon_config()
		else:
			self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)    

	def _create_visualizer(self, img, predictions):
		'''Perform inference.'''
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
		visualizer.output.scale = 1.0
		visualizer._default_font_size = 35
		visualizer.draw_instance_predictions(predictions['instances'].to('cpu'))
		return visualizer.output.get_image()

	def detect_objects(self, metadata=None, predictions=None):
		if not metadata:
			metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		instances = predictions['instances']
		detected_objects = []
		for i in range(len(instances)):
			class_id = instances.pred_classes[i].item()
			class_name = metadata.thing_classes[class_id]
			detected_objects.append(class_name)
		return detected_objects
	
	def window(self, full_screen = False):
		'''Screen configurations.'''
		cv2.namedWindow('result', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('result', 800, 600)
		if full_screen:
			cv2.setWindowProperty('result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			
	def _save_gif(self, gif_path: str, images: list):
		print('Saving gif...')
		save_dir = os.path.join(os.path.dirname(__file__), '..', gif_path)
		imageio.mimsave(save_dir, images, format='GIF', duration=1/30, loop=0)
		print(f'Gif saved at {save_dir}\n')

	def _process_frame(self, frame, predictor, visualizer):
		outputs = predictor(frame)
		v = visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
		return v.get_image()
    
	def _show_image(self, image_path):
		print('\nRunning inference on image...')
		try:
			# Load image using OpenCV
			img = cv2.imread(image_path)
			if img is None:
				raise ValueError('Failed to load image.')
			
			if self.model_weights: # Inference using model trained on custom dataset.				
				_metadata = register_dataset()
				test_dataset_dicts = get_data_dicts(data_path + 'test', class_names)
				image_dict = next((d for d in test_dataset_dicts if d["file_name"] == image_path), None)
				
				if image_dict is not None:
					img = cv2.imread(image_dict["file_name"])
					predictions = self.predictor(img)

					# Visualize the predicted image
					v_pred = Visualizer(img[:, :, ::-1], metadata=_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
					v_pred = v_pred.draw_instance_predictions(predictions["instances"].to("cpu"))
					plt.figure(figsize=(14, 10))
					plt.subplot(121)
					plt.imshow(cv2.cvtColor(v_pred.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
					plt.title("Predicted Image")

					# Load and visualize the ground truth image
					img_gt = cv2.imread(image_dict["file_name"])
					v_gt = Visualizer(img_gt[:, :, ::-1], metadata=_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
					v_gt = v_gt.draw_dataset_dict(image_dict)
					plt.subplot(122)
					plt.imshow(cv2.cvtColor(v_gt.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
					plt.title("Ground Truth Image")
					
					plt.tight_layout()
					plt.show()
					detected_objects = self.detect_objects(_metadata, predictions)
	
			else: # Inference using built-in model.
				# Process image using predictor
				predictions = self.predictor(img)
				image_to_show = self._create_visualizer(img, predictions)
				detected_objects = self.detect_objects(predictions=predictions)
				self.window()
				cv2.imshow('result', image_to_show[:, :, ::-1])
				while True:
					key = cv2.waitKey(1)
					# Check for ESC or 'q' key of button click on the GUI's X icon:
					if key == 27 or key == ord('q') or cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1: 
						break
				cv2.destroyAllWindows()
			# Print detected objects.
			self.count_items(detected_objects)
		except Exception as e:
			print("Error processing image:", e)
		
	def _show_video(self, video_path, save_gif, save_video, skip_frames, frame_batch, res_factor, verbose):
		print('\nRunning inference on video...')
		skip_frames = skip_frames or 1
		frame_count = 0
		used_frames = 0
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise ValueError('Failed to load video file.')

		if save_video:
			percentage = float(res_factor) if res_factor else 1
			_metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0] if len(self.cfg.DATASETS.TRAIN) else "__unused")
			visualizer = VideoVisualizer(_metadata)
			video = VideoFileClip(video_path)
			output_video = video.fl_image(lambda frame: self._process_frame(frame, self.predictor, visualizer))
			output_video = output_video.resize(height=int(video.size[1] * percentage))
			output_video.write_videofile(save_video, audio=True)
			
		elif save_gif:
			frames = []	
			fig, ax = plt.subplots()			
			#plt.ioff() # Switch off the interactive mode.
			predicting = True
			print('\nPredicting frames... This may take a while...')
			while predicting:
				ret, image = cap.read()
				if not ret:
					break
				frame_count += 1
				if frame_count % skip_frames != 0:
					continue
				used_frames +=1
				if res_factor:
					height, width = image.shape[:2]
					reduction_factor = 1 - res_factor / 100
					new_width = int(width * reduction_factor)
					new_height = int(height * reduction_factor)
					image = cv2.resize(image, (new_width, new_height))
				predictions =  self.predictor(image)
				image_to_show = self._create_visualizer(image, predictions)
				detected_objects = self.detect_objects(predictions=predictions)
				self.count_items(detected_objects)
				if verbose:
					ax.imshow(image_to_show)
					ax.axis('off')
					plt.pause(.2)
					plt.draw()
				if frame_batch:
					if used_frames == frame_batch:
						predicting = False	
				frames.append(image_to_show)
			plt.close(fig)
			self._save_gif(gif_path=save_gif, images=frames)		

	def _show_webcam(self):		
		print('\nRunning inference on webcam...')
		cap = cv2.VideoCapture(0)
		while True:
			ret, image = cap.read()
			if not ret:
				break
			predictions =  self.predictor(image)
			image = self._create_visualizer(image, predictions)
			detected_objects = self.detect_objects(predictions=predictions)
			print(detected_objects)
			self.window()
			#cv2.imshow('Image', image_to_show[:, :, ::-1])
			cv2.imshow("result", image)
			key = cv2.waitKey(1) & 0xFF
			if key == 27 or key == ord('q') or cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1: 
				break
		cv2.destroyAllWindows()
		cap.release()

	def count_items(self, objects: list):
		counts = Counter(objects)
		print('\n')
		for string, count in counts.items():
				print(f'Number of {string}s detected: {count}') 
		
	def main(self, cam, image_name, video_name, save_gif, save_video, skip_frames, frame_batch, res_factor, verbose):
		base_path = os.path.join(os.path.dirname(__file__), '..')
		if image_name:
			file_path = os.path.join(base_path, image_name)
			self._show_image(file_path)
		elif video_name:
			file_path = os.path.join(base_path, video_name)
			self._show_video(file_path, save_gif, save_video, skip_frames, frame_batch, res_factor, verbose)
		elif cam:
			self._show_webcam()	
				
if __name__ == '__main__':
	detectron = Detectron(model_type=args.model, model_weights=args.weights, threshold=args.threshold, device=args.device)
	detectron.main(args.cam, args.image, args.video, args.save_gif, args.save_video, args.skip_frames, args.frame_batch, args.res_factor, args.verbose)
