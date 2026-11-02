"""Evaluate the bundled model on raw test images and check parity with training-time logits.

python scripts/evaluate.py --limit 150
"""

import sys

from acpolyp.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
