import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


def _load_prediction_refiner():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "segmentation" / "prediction_refiner.py"
    spec = importlib.util.spec_from_file_location("prediction_refiner_disambiguation_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _draw_left_paren():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.ellipse(image, (15, 14), (max(2, 16 // 6), max(7, 16 // 2)), 0, 78, 282, 255, 2)
    return image


def _draw_right_paren():
    return cv2.flip(_draw_left_paren(), 1)


def _draw_slash():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(image, (8, 22), (20, 6), 255, 2)
    return image


def _draw_one():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(image, (13, 4), (13, 24), 255, 2)
    cv2.line(image, (10, 8), (13, 5), 255, 2)
    cv2.line(image, (10, 24), (16, 24), 255, 2)
    return image


def _draw_plus():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(image, (7, 14), (21, 14), 255, 2)
    cv2.line(image, (14, 7), (14, 21), 255, 2)
    return image


def _draw_minus():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(image, (7, 14), (21, 14), 255, 2)
    return image


def _draw_times():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(image, (8, 8), (20, 20), 255, 2)
    cv2.line(image, (20, 8), (8, 20), 255, 2)
    return image


def _draw_digit(char):
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(image, char, (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 255, 2, cv2.LINE_AA)
    return image


def _draw_char(char):
    if char == "(":
        return _draw_left_paren()
    if char == ")":
        return _draw_right_paren()
    if char == "/":
        return _draw_slash()
    if char == "1":
        return _draw_one()
    if char == "+":
        return _draw_plus()
    if char == "-":
        return _draw_minus()
    if char == "*":
        return _draw_times()
    return _draw_digit(char)


def _make_top_k(*entries):
    return [
        {"char": char, "conf": conf}
        for char, conf in entries
    ]


def _build_expression_case(specs):
    rects = []
    rois = []
    raw_predictions = []
    cursor_x = 0

    for spec in specs:
        shape_char = spec["shape"]
        predicted_char = spec.get("pred", shape_char)
        confidence = spec.get("conf", 0.99 if predicted_char == shape_char else 0.56)
        top_k = spec.get("top_k", _make_top_k((predicted_char, confidence)))
        roi = _draw_char(shape_char)

        rects.append((cursor_x, 0, roi.shape[1], roi.shape[0]))
        rois.append(roi)
        raw_predictions.append({
            "char": predicted_char,
            "conf": confidence,
            "raw_char": predicted_char,
            "raw_conf": confidence,
            "top_k": top_k,
        })
        cursor_x += roi.shape[1] + 8

    return rects, rois, raw_predictions


class PredictionRefinerDisambiguationTests(unittest.TestCase):
    def setUp(self):
        self.prediction_refiner = _load_prediction_refiner()

    def _assert_refined_expression(self, specs, expected):
        rects, rois, raw_predictions = _build_expression_case(specs)
        refined = self.prediction_refiner.refine_predictions(rects, rois, raw_predictions)
        self.assertEqual(expected, "".join(item["char"] for item in refined))

    def test_slash_label_with_one_shape_becomes_one(self):
        image = _draw_one()
        features = self.prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
        item = {
            "char": "/",
            "raw_char": "/",
            "conf": 0.56,
            "raw_conf": 0.56,
            "top_k": _make_top_k(
                ("/", 0.56),
                ("1", 0.41),
                ("(", 0.02),
                (")", 0.01),
            ),
            "features": features,
        }

        self.assertTrue(self.prediction_refiner._looks_like_one(features))
        self.assertFalse(self.prediction_refiner._looks_like_open_boundary(features))
        self.assertFalse(self.prediction_refiner._looks_like_close_boundary(features))
        self.assertEqual("1", self.prediction_refiner._override_char(item, prev_char=None, next_char="5"))

    def test_slash_label_with_open_paren_shape_becomes_open_paren(self):
        image = _draw_left_paren()
        features = self.prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
        item = {
            "char": "/",
            "raw_char": "/",
            "conf": 0.54,
            "raw_conf": 0.54,
            "top_k": _make_top_k(
                ("/", 0.54),
                ("(", 0.33),
                ("1", 0.09),
                (")", 0.03),
            ),
            "features": features,
        }

        self.assertTrue(self.prediction_refiner._looks_like_open_boundary(features))
        self.assertEqual("(", self.prediction_refiner._override_char(item, prev_char=None, next_char="6"))

    def test_open_paren_label_with_slash_shape_becomes_slash(self):
        image = _draw_slash()
        features = self.prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
        item = {
            "char": "(",
            "raw_char": "(",
            "conf": 0.57,
            "raw_conf": 0.57,
            "top_k": _make_top_k(
                ("(", 0.57),
                ("/", 0.38),
                ("1", 0.03),
                (")", 0.02),
            ),
            "features": features,
        }

        self.assertTrue(self.prediction_refiner._looks_like_slash(features))
        self.assertEqual("/", self.prediction_refiner._override_char(item, prev_char="5", next_char="6"))

    def test_one_label_with_slash_shape_becomes_slash_in_term_context(self):
        image = _draw_slash()
        features = self.prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
        item = {
            "char": "1",
            "raw_char": "1",
            "conf": 0.998,
            "raw_conf": 0.998,
            "top_k": _make_top_k(
                ("1", 0.998),
                ("/", 0.002),
            ),
            "features": features,
        }

        self.assertTrue(self.prediction_refiner._looks_like_slash(features))
        self.assertEqual("/", self.prediction_refiner._override_char(item, prev_char="8", next_char="3"))

    def test_plus_label_with_x_support_becomes_star_in_term_context(self):
        image = _draw_times()
        features = self.prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
        item = {
            "char": "+",
            "raw_char": "+",
            "conf": 0.57,
            "raw_conf": 0.57,
            "top_k": _make_top_k(
                ("+", 0.57),
                ("*", 0.40),
                ("7", 0.02),
            ),
            "features": features,
        }

        self.assertEqual("*", self.prediction_refiner._override_char(item, prev_char="1", next_char="7"))

    def test_confident_minus_fragment_is_not_dropped_as_noise(self):
        image = _draw_minus()
        rect = (0, 0, 16, 4)
        features = self.prediction_refiner._extract_features(image, rect)
        item = {
            "char": "-",
            "raw_char": "-",
            "conf": 0.999,
            "raw_conf": 0.999,
            "top_k": _make_top_k(
                ("-", 0.999),
                ("/", 0.001),
            ),
            "features": features,
        }

        self.assertTrue(self.prediction_refiner._looks_like_minus(features))
        self.assertFalse(self.prediction_refiner._should_drop_noise(item, median_area=1200.0, median_height=45.0))

    def test_attached_case_paren_one_slash_sequence_refines(self):
        specs = [
            {"shape": "(", "pred": "/", "top_k": _make_top_k(("/", 0.54), ("(", 0.35), ("1", 0.08), (")", 0.03))},
            {"shape": "1", "pred": "/", "top_k": _make_top_k(("/", 0.56), ("1", 0.41), ("(", 0.02), (")", 0.01))},
            {"shape": "*"},
            {"shape": "7"},
            {"shape": ")", "pred": "/", "top_k": _make_top_k(("/", 0.55), (")", 0.33), ("1", 0.08), ("(", 0.04))},
            {"shape": "*"},
            {"shape": "3"},
            {"shape": "+"},
            {"shape": "5"},
            {"shape": "/", "pred": "(", "top_k": _make_top_k(("(", 0.56), ("/", 0.38), ("1", 0.04), (")", 0.02))},
            {"shape": "6"},
        ]
        self._assert_refined_expression(specs, "(1*7)*3+5/6")

    def test_attached_case_nested_group_with_division_refines(self):
        specs = [
            {"shape": "7"},
            {"shape": "/", "pred": ")", "top_k": _make_top_k((")", 0.55), ("/", 0.37), ("1", 0.05), ("(", 0.03))},
            {"shape": "3"},
            {"shape": "+"},
            {"shape": "(", "pred": "1", "top_k": _make_top_k(("1", 0.58), ("(", 0.30), ("/", 0.08), (")", 0.04))},
            {"shape": "5"},
            {"shape": "-"},
            {"shape": "6"},
            {"shape": "+"},
            {"shape": "2"},
            {"shape": ")", "pred": "1", "top_k": _make_top_k(("1", 0.57), (")", 0.34), ("/", 0.05), ("(", 0.04))},
            {"shape": "*"},
            {"shape": "2"},
        ]
        self._assert_refined_expression(specs, "7/3+(5-6+2)*2")

    def test_attached_case_group_then_division_refines(self):
        specs = [
            {"shape": "(", "pred": "1", "top_k": _make_top_k(("1", 0.58), ("(", 0.31), ("/", 0.07), (")", 0.04))},
            {"shape": "5"},
            {"shape": "+"},
            {"shape": "3"},
            {"shape": "-"},
            {"shape": "2"},
            {"shape": ")", "pred": "/", "top_k": _make_top_k(("/", 0.54), (")", 0.34), ("1", 0.07), ("(", 0.05))},
            {"shape": "*"},
            {"shape": "8"},
            {"shape": "/", "pred": ")", "top_k": _make_top_k((")", 0.55), ("/", 0.36), ("1", 0.05), ("(", 0.04))},
            {"shape": "3"},
        ]
        self._assert_refined_expression(specs, "(5+3-2)*8/3")

    def test_attached_case_division_then_group_refines(self):
        specs = [
            {"shape": "4"},
            {"shape": "+"},
            {"shape": "5"},
            {"shape": "-"},
            {"shape": "6"},
            {"shape": "/", "pred": "(", "top_k": _make_top_k(("(", 0.56), ("/", 0.38), ("1", 0.04), (")", 0.02))},
            {"shape": "7"},
            {"shape": "*"},
            {"shape": "3"},
            {"shape": "+"},
            {"shape": "(", "pred": "1", "top_k": _make_top_k(("1", 0.58), ("(", 0.31), ("/", 0.07), (")", 0.04))},
            {"shape": "5"},
            {"shape": "-"},
            {"shape": "4"},
            {"shape": ")", "pred": "/", "top_k": _make_top_k(("/", 0.54), (")", 0.34), ("1", 0.07), ("(", 0.05))},
        ]
        self._assert_refined_expression(specs, "4+5-6/7*3+(5-4)")


if __name__ == "__main__":
    unittest.main()
