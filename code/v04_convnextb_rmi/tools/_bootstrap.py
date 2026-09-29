"""Make the sibling MMSeg checkout and this project importable."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = PROJECT_ROOT.parent / 'mmsegmentation'


def bootstrap():
    for path in (PROJECT_ROOT, MMSEG_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.chdir(PROJECT_ROOT)
