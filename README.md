# PatternRecognitionGroupProject

## Installation

Install any recent version of Python, tested on Python 3.8.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to use

If you only want to greyscale the `CelebA` dataset, run the `greyscaling218x178.py` file. This will generate folders with training, validation and test data. This script will preserve the original resolution and aspect ratio.

If you want to change the resolution to 256x256 and add zero-padding to keep the aspect ratio, run the `greyscaling256x256.py` script. This script might be useful for certain models that require a specific input.

If you want to convert a padded image (256x256) to its original resolution (218x178) and aspect ratio, use the `padded2original.py` script.
