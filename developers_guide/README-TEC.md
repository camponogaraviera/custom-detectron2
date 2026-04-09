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
└── README-TESTS.md

src/
├── __init__.py
├── inference.py
├── register.py
├── settings.py
├── train.py
├── utils/
│   ├── __init__.py
│   ├── parse_int_tuple.py
│   └── resolve_config_path.py
└── video_output.py

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

- `PyTorch`: deep learning framework used to build and fine-tune the pre-trained model.
- `pytorch-cuda`: provides CUDA runtime support so PyTorch can utilize NVIDIA GPUs for acceleration.
- `pytorch-mutex`: ensures only one PyTorch variant (CPU or CUDA) is installed to avoid conflicts.
- `torchtriton`: backend compiler used by PyTorch for optimizing and generating efficient GPU kernels.
- `torchvision`: provides datasets, pre-trained models, and image transformations for computer vision tasks.
- `cython`: used to compile Python-like code into C for performance improvements.
- `moviepy`: video editing library for creating, processing, and manipulating video files in Python.
- `opencv`: library for image and video processing (e.g., resizing, augmentation, I/O, visualization).
- `imageio`: library for reading and writing image and video files in various formats.
- `numpy`: core library for numerical operations and array manipulation.
- `pytest`: framework for writing and running unit tests.
- `black`: opinionated code formatter for consistent Python style.
- `flake8`: linting tool for enforcing style and catching errors.
- `pre-commit`: framework for managing Git pre-commit hooks to enforce code quality checks before commits.
