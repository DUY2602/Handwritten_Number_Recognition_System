import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segmentation.image_ops import _extract_horizontal_line_mask, _preserve_short_horizontal_strokes
from segmentation.segmentation import segment_image


class NotebookOperatorPreservationTests(unittest.TestCase):
    def test_short_horizontal_stroke_is_preserved_while_ruling_line_is_removed(self):
        height, width = 120, 300
        binary = np.zeros((height, width), dtype=np.uint8)
        cv2.line(binary, (0, 40), (width - 1, 40), 255, 1)
        cv2.line(binary, (0, 80), (width - 1, 80), 255, 1)
        cv2.line(binary, (90, 40), (120, 40), 255, 4)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(width * 0.25)), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        h_lines = cv2.max(
            h_lines,
            _extract_horizontal_line_mask(
                binary,
                min_span_ratio=0.28,
                bridge_gap_ratio=0.05,
                max_line_height_ratio=0.10,
                lower_bias=False,
            ),
        )

        preserved = _preserve_short_horizontal_strokes(binary, h_lines, width, height)
        cleaned = cv2.subtract(binary, preserved)

        self.assertGreater(np.count_nonzero(cleaned[34:47, 84:126]), 80)
        self.assertEqual(0, np.count_nonzero(cleaned[80]))

    def test_upload_segmentation_keeps_minus_boxes_on_notebook_row(self):
        height, width = 320, 900
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        for y in range(60, height, 60):
            cv2.line(image, (0, y), (width - 1, y), (180, 180, 180), 2)

        ink_color = (160, 60, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "8", (40, 140), font, 2.4, ink_color, 6, cv2.LINE_AA)
        cv2.line(image, (130, 120), (170, 120), ink_color, 6)
        cv2.putText(image, "2", (190, 140), font, 2.4, ink_color, 6, cv2.LINE_AA)
        cv2.line(image, (280, 145), (320, 95), ink_color, 6)
        cv2.putText(image, "4", (340, 140), font, 2.4, ink_color, 6, cv2.LINE_AA)
        cv2.line(image, (440, 120), (480, 120), ink_color, 6)
        cv2.putText(image, "5", (500, 140), font, 2.4, ink_color, 6, cv2.LINE_AA)

        tmp_dir = ROOT_DIR / ".tmp_tests"
        tmp_dir.mkdir(exist_ok=True)
        image_path = tmp_dir / "notebook_minus_case.png"
        cv2.imwrite(str(image_path), image)
        try:
            _, rects, _, _ = segment_image(str(image_path), debug=False, input_mode="upload")
        finally:
            if image_path.exists():
                image_path.unlink()

        dash_like = [rect for rect in rects if rect[2] >= 25 and rect[3] <= 12]
        self.assertGreaterEqual(len(rects), 7)
        self.assertEqual(2, len(dash_like))


if __name__ == "__main__":
    unittest.main()
