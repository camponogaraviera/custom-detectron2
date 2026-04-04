<!-- Logos -->

<a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer"><img src="https://github.com/camponogaraviera/logos/blob/main/assets/pytorch.png" width="110"></a>

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.9.13-informational)](https://www.python.org/downloads/source/)
[![Torch](https://img.shields.io/badge/Torch-2.2.0-%23EE4C2C)](https://pytorch.org/)

> Originally implemented in 2022. Minor cleanup in 2026.

<!-- Title -->
<div align='center'>
  <h1> Object Detection and Image Segmentation Pipeline </h1>
  <h2> Fine-tuning Detectron2 Models on a Custom Dataset </h2>
</div>

# About

- Demonstrates how to fine-tune [Detectron2](https://github.com/facebookresearch/detectron2) models (Mask R-CNN, Faster R-CNN) for image segmentation and object detection on a [custom](https://detectron2.readthedocs.io/tutorials/datasets.html) dataset.
- Provides an image segmentation dataset manually built using Labelme+ for bounding box and mask annotations.
- MLOps pipeline: Data pre-processing and registration, model fine-tuning, evaluation, and real-time inference with automated tests.

Note: One can find a summary of object detection models at the end of this README.

# Results

<p align="center">
  <a href="assets/inference_tissue1.png"><img src="assets/inference_tissue1.png" width="40%"></a>
  <a href="assets/inference_tissue2.png"><img src="assets/inference_tissue2.png" width="40%"></a>
</p>

<p align="center">
  <a href="assets/inference_cat.png"><img src="assets/inference_cat.png" width="40%"></a>
  <a href="assets/gif.gif"><img src="assets/gif.gif" width="40%"></a> 
</p>

https://github.com/user-attachments/assets/de48d6b1-0867-446d-a336-666a40493a8e

---

<!-- #region Project Architecture & Technology Stack -->
<details>
  <summary><h1 id="tech">Project Architecture & Technology Stack</h1></summary>

See [README-TEC.md](developers_guide/README-TEC.md).

</details>
<!-- #endregion -->

---

<!-- #region Conda Environment -->
<details>
  <summary><h1 id="">Conda Environment</h1></summary>

Create the conda environment:

```bash
conda env create -f environment.yml --verbose && conda activate detectron2
```

Install detectron2 (make sure to have `gcc & g++ ≥ 5.4`).

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## labelme+

Pip install `labelme+` to manually label images for your custom dataset. 
To avoid conflicts, create a new conda environment:

```bash
conda create -yn labelme python==3.9.13 && conda activate labelme
```

```bash
pip install enhancedlabelme==1.8.2
```

Run from terminal:

```bash
labelme+
```

> Note: Do not rename the image files after labeling, otherwise the '.json' file metadata will lose its link with the image.

</details>
<!-- #endregion -->

---

<!-- #region Training Usage -->
<details>
  <summary><h1 id="">Training Usage</h1></summary>

```ShellSession
train.py <model_type> <device> <config_file>

Required Arguments:
  model_type                       Model type: 'object_detection', 'instance_segmentation', 'panoptic_segmentation', or 'keypoint'.

Optional Arguments:
  device                           Training with either 'cpu' or 'cuda'. Defaults to 'cpu'.
  config_file                      Path of the configuration file containing hyperparameters. Defaults to 'configs/config.cfg'.
```

Usage:

```bash
python -m src.train 'instance_segmentation'
```

</details>
<!-- #endregion -->

---

<!-- #region Inference Usage -->
<details>
  <summary><h1 id="">Inference Usage</h1></summary>

```ShellSession
inference.py --model [--threshold] [--image] [--video] [--cam] [--skip_frames] [--frame_batch]
[--res_factor] [--verbose] [--save_gif] [--save_video] [--max_video_size_mb]

Required Arguments:
  -m, --model (str)                       Model type: 'object_detection', 'instance_segmentation', 'panoptic_segmentation', or 'keypoint'.

Optional Arguments:
  -w, --weights (str)                     Path to the model's weights.
  -t, --threshold (float)                 Detection threshold value.
  -d, --device (str)                      Device option ('cpu' or 'gpu'). Defaults to 'cpu'.
  -i, --image (str)                       Filename to run inference on image.
  -v, --video (str)                       Filename to run inference on video.
  -c, --cam (bool)                        Whether to run inference on webcam.
  -sg, --save_gif (str)                   Path to save a gif of video inference.
  -sv, --save_video (str)                 Path to save a video of video inference.
  --max_video_size_mb (float)             Maximum size in MB for saved videos. Defaults to 10. Use 0 to disable recompression.
  -vb, --verbose (bool)                   Whether to display the gif inference on-the-fly.
  -sf, --skip_frames (int)                Number of frames to skip between detections.
  -fb, --frame_batch (int)                Number of total frames to be detected.
  -rf, --res_factor (float)               Percentage by which the output height is reduced while preserving aspect ratio. Defaults to 0.
```

Run inference on an image with a custom model:

```bash
python -m src.inference -m 'instance_segmentation' -w '../artifacts/model_final.pth' --image 'dataset/test/benign_2.jpg'
```

Run inference on an image with the built-in model:

```bash
python -m src.inference -m 'panoptic_segmentation' -t 0.8 --image 'assets/cat.png'
```

Run inference on video and save GIF:

```bash
python -m src.inference -m 'instance_segmentation' -t 0.8 --video 'assets/taipei.mp4' --save_gif 'assets/gif.gif' --skip_frames 10 --frame_batch 500
```

Run inference on video and save video:

```bash
python -m src.inference -m 'instance_segmentation' -t 0.5 --video 'assets/taipei.mp4' --save_video 'assets/taipei_inference.mp4' --max_video_size_mb 10
```

Run inference on webcam:

```bash
python -m src.inference -m 'panoptic_segmentation' --cam True
```

</details>
<!-- #endregion -->

---

<!-- #region Evolution of object detection models -->
<details>
  <summary><h1 id="">Evolution of Detection Models</h1></summary>

- [R-CNN](https://arxiv.org/abs/1311.2524): computes about 2k region proposals per input image, after which each of these region proposals is then individually passed through a CNN. As a result, there will be a total of 2k passes through the CNN (although in parallel it is computationally expensive). It uses selective search as the external proposal method to find the region proposals.
- [Fast R-CNN](https://arxiv.org/abs/1504.08083): each input image is first passed through a CNN, after which region proposals are computed. As a result, there is only a single pass through the CNN per image instead of 2k passes as in R-CNN. It also uses selective search to find the region proposals, however, it introduced the RoI (Region of Interest) pooling layer.
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) = Fast R-CNN+RPN. The difference is that now the region proposals are computed with a region proposal network (RPN). Region proposals are predicted at each sliding window over the output convolutional feature map. Anchor boxes with different scale and aspect ratio are then centered at the anchor point of each sliding window location. Each sliding window is then mapped to a lower dimensional feature that is fed into two fully-connected layers, one for regression (to predict bounding box coordinates) and the other for classification (to distinguish between object and background). The idea of the regression loss is to minimize the dissimilarity between the predicted bounding box and the ground truth box.
- [YOLO](https://arxiv.org/abs/1506.02640): is a single-shot neural network for object detection that divides an input image into an SxS grid and uses deep CNNs. There is no region proposal network. The model predicts multiple bounding boxes and corresponding class probabilities for each grid.
- [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT): is a neural network model based on the "[attention is all you need](https://arxiv.org/abs/1706.03762)" Transformer architecture that can be applied to [image classification](https://keras.io/examples/vision/image_classification_with_vision_transformer/) and [object recognition](https://keras.io/examples/vision/object_detection_using_vision_transformer/) tasks. I wrote a quick summary [here](https://github.com/camponogaraviera/vision-transformer).

- Stats (results may vary according to hardware specs, image size, and neural network):
  - Faster R-CNN: 5-17 FPS.
  - YOLO: 40-91 FPS.

## Terminology

- Region proposal: a region that might contain an object of interest.
- Region of interest: selected bounding boxes from the region proposal step.
- IoU: is the Intersection over Union between the bounding box (built with the anchor points and aspect ratios) and the ground truth box.
- Anchor point: predefined locations or points in an image grid where anchor boxes are placed. Each anchor point is associated with multiple anchor boxes, which vary in scale and aspect ratio to handle objects of different shapes and sizes in the image.
- Anchor box: is a helper/reference (initial guess) used to generate a bounding box during training. An anchor box is used for training if it satisfies two conditions:
  - IoU > .7 is positive (there is an object).
  - IoU < .3 is negative (there is no object).
- Bounding box: is a fine-tuned anchor box represented as a rectangle with four coordinates that enclose the object of interest. During training, the model predicts bounding box coordinates and class probabilities to localize and classify objects within the image, respectively.

</details>
<!-- #endregion -->
