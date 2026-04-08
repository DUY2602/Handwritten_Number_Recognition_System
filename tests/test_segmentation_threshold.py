import math
import os
import sys
import unittest

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from segmentation.image_ops import _score_threshold


class ScoreThresholdTests(unittest.TestCase):
    def test_empty_binary_returns_negative_infinity(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        self.assertTrue(math.isinf(_score_threshold(binary)))
        self.assertLess(_score_threshold(binary), 0)

    def test_character_like_blobs_score_higher_than_line_dominated_mask(self):
        char_like = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(char_like, (14, 16), (22, 48), 255, thickness=-1)
        cv2.rectangle(char_like, (34, 16), (48, 48), 255, thickness=-1)

        line_dominated = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(line_dominated, (5, 30), (59, 33), 255, thickness=-1)

        self.assertGreater(_score_threshold(char_like), _score_threshold(line_dominated))

    def test_centered_foreground_scores_higher_than_edge_heavy_mask(self):
        centered = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(centered, (24, 32), 8, 255, thickness=-1)
        cv2.circle(centered, (42, 32), 8, 255, thickness=-1)

        edge_heavy = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(edge_heavy, (0, 0), (63, 3), 255, thickness=-1)
        cv2.rectangle(edge_heavy, (0, 0), (3, 63), 255, thickness=-1)

        self.assertGreater(_score_threshold(centered), _score_threshold(edge_heavy))


if __name__ == "__main__":
    unittest.main()
