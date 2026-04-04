<div align='center'>
  <h1> Project Architecture & Technology Stack </h1>
</div>

# Table of Contents

- [File Structure Tree](#file-structure-tree)
- [Dependencies](#dependencies)

# File Structure Tree

```bash
artifacts/

assets/

configs/
└── config.cfg

dataset/
│   ├── test/
│   └── train/

developers_guide/
├── README-TEC.md
└── README-Tests.md

src/
├── __init__.py
├── inference.py
├── register.py	
├── train.py
└── utils.py

tests/
├── conftest.py
└── test_inference.py

.editorconfig
.flake8
.gitattributes
.gitignore
.pre-commit-config.yaml
README.md
environment.yml
```

---

# Dependencies

- `PyTorch`: Deep learning framework used to build and fine-tune the pre-trained model.
- `pytorch-cuda`: Provides CUDA runtime support so PyTorch can utilize NVIDIA GPUs for acceleration.
- `pytorch-mutex`: Ensures only one PyTorch variant (CPU or CUDA) is installed to avoid conflicts.
- `torchtriton`: Backend compiler used by PyTorch for optimizing and generating efficient GPU kernels.
- `torchvision`: Provides datasets, pre-trained models, and image transformations for computer vision tasks.
- `cython`: Used to compile Python-like code into C for performance improvements.
- `moviepy`: Video editing library for creating, processing, and manipulating video files in Python.
- `opencv`: Library for image and video processing (e.g., resizing, augmentation, I/O, visualization).
- `imageio`: Library for reading and writing image and video files in various formats.
- `numpy`: Core library for numerical operations and array manipulation.
- `pytest`: Framework for writing and running unit tests.
- `black`: Opinionated code formatter for consistent Python style.
- `flake8`: Linting tool for enforcing style and catching errors.
- `pre-commit`: Framework for managing Git pre-commit hooks to enforce code quality checks before commits.
