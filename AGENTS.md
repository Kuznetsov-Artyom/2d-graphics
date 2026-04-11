# Repository Structure

This is a 2D graphics/image processing coursework repository with Jupyter notebooks organized by lab assignments.

## Directories

- `lab1/` - Image processing homework (gamma correction, histogram analysis, thresholding)
- `lab2/` - Advanced image processing (patch extraction, filtering)
- `lab3/` - Lecture materials and documentation

## Environment Setup

Lab1 has a Python virtual environment with dependencies in `lab1/requirements.txt`:

```bash
cd lab1
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Key dependencies: opencv-python, matplotlib, numpy, scikit-image, scipy

## Working with Notebooks

The notebooks are in Russian and contain image processing algorithms and homework assignments. Use Jupyter Lab or Jupyter Notebook to run them.