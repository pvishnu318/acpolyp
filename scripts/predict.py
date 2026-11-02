"""Predict one image with the bundled model: python scripts/predict.py frame.jpg --cam cam.png"""

import sys

from acpolyp.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    main()
