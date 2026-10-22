# OpenCV Contour Detection Mini Project

A reproducible image-processing demonstration: **grayscale → Gaussian blur → Canny edges → morphological closing → external contours → overlay**.

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Open `notebooks/01_contour_detection.ipynb` from the repository in VS Code or Jupyter and run all cells. It generates a synthetic example image in `images/` when needed. Results are written to `outputs/`.

## Test

```bash
python -m pytest -q
```

The sample uses simple geometric shapes to make contour outputs reproducible. This is a learning demo; threshold choices and object/background contrast affect results.
