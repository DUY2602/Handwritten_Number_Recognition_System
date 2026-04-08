import os
import sys

import cv2
import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

OUTPUT_ROOT = os.path.join(CURRENT_DIR, "output_digit")
DEBUG_ROOT = os.path.join(CURRENT_DIR, "output_debug")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(DEBUG_ROOT, exist_ok=True)


def read_image_safe(path):
    try:
        with open(path, "rb") as f:
            data = bytearray(f.read())
        return cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        logger.warning("Failed to read image '%s': %s", path, exc)
        return None


def _clean_dir(directory):
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

def _mode_artifact_dir(root_dir, input_mode):
    mode_dir = os.path.join(root_dir, input_mode)
    os.makedirs(mode_dir, exist_ok=True)
    return mode_dir
