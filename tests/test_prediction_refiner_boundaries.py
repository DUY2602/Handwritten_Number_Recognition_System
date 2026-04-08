import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _load_prediction_refiner():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "segmentation" / "prediction_refiner.py"
    spec = importlib.util.spec_from_file_location("prediction_refiner_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _draw_left_paren():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.ellipse(image, (15, 14), (max(2, 16 // 6), max(7, 16 // 2)), 0, 78, 282, 255, 2)
    return image


def _draw_right_paren():
    image = np.zeros((28, 28), dtype=np.uint8)
    cv2.ellipse(image, (13, 14), (max(2, 16 // 6), max(7, 16 // 2)), 0, -102, 102, 255, 2)
    return image


def _load_feedback_image(*parts):
    path = Path(__file__).resolve().parents[1].joinpath(*parts)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"Missing test image: {path}"
    return image


def _make_top_k(*entries):
    return [
        {"char": char, "conf": conf}
        for char, conf in entries
    ]


def test_parenthesis_shape_overrides_numeric_one():
    prediction_refiner = _load_prediction_refiner()

    left_features = prediction_refiner._extract_features(_draw_left_paren(), (0, 0, 28, 28))
    right_features = prediction_refiner._extract_features(_draw_right_paren(), (0, 0, 28, 28))

    assert prediction_refiner._looks_like_lparen(left_features)
    assert not prediction_refiner._looks_like_one(left_features)
    assert prediction_refiner._override_char(
        {"char": "1", "conf": 0.58, "features": left_features},
        prev_char=None,
        next_char="3",
    ) == "("

    assert prediction_refiner._looks_like_rparen(right_features)
    assert not prediction_refiner._looks_like_one(right_features)
    assert prediction_refiner._override_char(
        {"char": "1", "conf": 0.58, "features": right_features},
        prev_char="3",
        next_char=None,
    ) == ")"


def test_slanted_one_stays_one_and_not_parenthesis():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "corrections",
        "1",
        "20260407T020436_3052891a_char_002_pred-7_label-1.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))

    assert prediction_refiner._looks_like_one(features)
    assert not prediction_refiner._looks_like_lparen(features)
    assert not prediction_refiner._looks_like_rparen(features)
    assert prediction_refiner._map_numeric_char({"char": "(", "features": features}) == "1"
    assert prediction_refiner._override_char(
        {"char": "1", "conf": 0.62, "features": features},
        prev_char=None,
        next_char="5",
    ) == "1"


def test_square_bracketish_one_with_parenthesis_support_becomes_open_paren():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "analyses",
        "20260408T051605_3b3df75f",
        "characters",
        "char_000.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
    item = {
        "char": "1",
        "raw_char": "1",
        "conf": 0.718103,
        "raw_conf": 0.718103,
        "top_k": _make_top_k(
            ("1", 0.718103),
            ("(", 0.11256),
            ("5", 0.053484),
            ("8", 0.04707),
            ("4", 0.025796),
        ),
        "features": features,
    }

    assert prediction_refiner._looks_like_open_boundary(features)
    assert prediction_refiner._override_char(item, prev_char=None, next_char="6") == "("


def test_confident_six_without_parenthesis_support_stays_six():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "analyses",
        "20260408T051605_3b3df75f",
        "characters",
        "char_001.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
    item = {
        "char": "6",
        "raw_char": "6",
        "conf": 0.999946,
        "raw_conf": 0.999946,
        "top_k": _make_top_k(
            ("6", 0.999946),
            ("8", 0.000035),
            ("5", 0.000019),
            ("0", 0.0),
            ("9", 0.0),
        ),
        "features": features,
    }
    candidate_scores = prediction_refiner._candidate_char_scores(item)

    assert not prediction_refiner._looks_like_open_boundary(features)
    assert prediction_refiner._override_char(item, prev_char=None, next_char="3") == "6"
    assert "(" not in candidate_scores


def test_slash_like_one_with_near_top_support_becomes_slash():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "analyses",
        "20260408T051605_3b3df75f",
        "characters",
        "char_006.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
    item = {
        "char": "1",
        "raw_char": "1",
        "conf": 0.532278,
        "raw_conf": 0.532278,
        "top_k": _make_top_k(
            ("1", 0.532278),
            ("/", 0.467559),
            ("5", 0.000055),
            ("9", 0.000032),
            ("7", 0.000028),
        ),
        "features": features,
    }

    assert prediction_refiner._looks_like_slash(features)
    assert prediction_refiner._override_char(item, prev_char="5", next_char="6") == "/"


def test_draw_mode_open_paren_like_one_becomes_open_paren():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "analyses",
        "20260408T053350_62705004",
        "characters",
        "char_000.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
    item = {
        "char": "1",
        "raw_char": "1",
        "conf": 0.918,
        "raw_conf": 0.918,
        "top_k": _make_top_k(
            ("1", 0.918),
            ("(", 0.078),
            ("/", 0.002),
            ("4", 0.001),
            ("9", 0.001),
        ),
        "features": features,
    }

    assert prediction_refiner._looks_like_open_boundary(features)
    assert prediction_refiner._override_char(item, prev_char=None, next_char="6") == "("


def test_draw_mode_open_paren_like_six_without_support_stays_six():
    prediction_refiner = _load_prediction_refiner()
    image = _load_feedback_image(
        "src",
        "model",
        "artifacts",
        "feedback",
        "analyses",
        "20260408T053358_7da0c62a",
        "characters",
        "char_000.png",
    )
    features = prediction_refiner._extract_features(image, (0, 0, image.shape[1], image.shape[0]))
    item = {
        "char": "6",
        "raw_char": "6",
        "conf": 0.706,
        "raw_conf": 0.706,
        "top_k": _make_top_k(
            ("6", 0.706),
            ("8", 0.179),
            ("1", 0.044),
            ("4", 0.022),
            ("5", 0.018),
        ),
        "features": features,
    }

    assert prediction_refiner._override_char(item, prev_char=None, next_char="6") == "6"
