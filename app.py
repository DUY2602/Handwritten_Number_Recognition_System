import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Add src to the module search path so we can import project files.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing.preprocessing import normalize_binary_character
from segmentation.context_corrector import build_raw_predictions
from segmentation.operator_classifier import predict_characters_top_k
from segmentation.expression_parser import build_and_evaluate
from segmentation.prediction_refiner import refine_predictions_by_line
from segmentation.segmentation import segment_image

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
FEEDBACK_ROOT = PROJECT_ROOT / "src" / "model" / "artifacts" / "feedback"
ANALYSES_DIR = FEEDBACK_ROOT / "analyses"
CORRECTIONS_DIR = FEEDBACK_ROOT / "corrections"
REJECTIONS_DIR = FEEDBACK_ROOT / "rejections"
REPORTS_DIR = FEEDBACK_ROOT / "reports"
FEEDBACK_LOG_PATH = FEEDBACK_ROOT / "feedback_log.jsonl"

for directory in (
    UPLOAD_FOLDER,
    FEEDBACK_ROOT,
    ANALYSES_DIR,
    CORRECTIONS_DIR,
    REJECTIONS_DIR,
    REPORTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

LINE_COLORS = [
    (15, 118, 110),
    (180, 83, 9),
    (29, 78, 216),
    (153, 27, 27),
]

CORRECTABLE_LABELS = set("0123456789+-*/()")
LABEL_ALIASES = {
    "plus": "+",
    "minus": "-",
    "mul": "*",
    "times": "*",
    "x": "*",
    "X": "*",
    "div": "/",
    "slash": "/",
    "forward_slash": "/",
    "÷": "/",
    "lparen": "(",
    "rparen": ")",
}
SAFE_LABEL_NAMES = {
    "+": "plus",
    "-": "minus",
    "*": "mul",
    "/": "div",
    "(": "lparen",
    ")": "rparen",
}


def _asset_version(relative_path):
    path = PROJECT_ROOT / relative_path
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def _utc_now():
    return datetime.now(timezone.utc)


def _utc_iso(timestamp=None):
    current = timestamp or _utc_now()
    return current.isoformat().replace("+00:00", "Z")


def _new_analysis_id(timestamp=None):
    current = timestamp or _utc_now()
    return f"{current.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"


def _serialize_rect(rect):
    return [int(value) for value in rect]


def _normalize_feedback_label(raw_label):
    label = str(raw_label or "").strip()
    if not label:
        return ""
    return LABEL_ALIASES.get(label, label)


def _safe_label_token(raw_label):
    normalized = _normalize_feedback_label(raw_label)
    if not normalized:
        return "unknown"
    return SAFE_LABEL_NAMES.get(normalized, normalized if normalized.isalnum() else "unknown")


def _encode_png_data_uri(image):
    if image is None or getattr(image, "size", 0) == 0:
        return ""

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_character_roi(thresh, rect, padding=4):
    if thresh is None:
        return np.zeros((28, 28), dtype=np.uint8)

    x, y, w, h = (int(value) for value in rect)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(thresh.shape[1], x + w + padding)
    y1 = min(thresh.shape[0], y + h + padding)
    roi = thresh[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    try:
        return normalize_binary_character(roi)
    except Exception:
        return cv2.resize(roi, (28, 28), interpolation=cv2.INTER_NEAREST)


def _serialize_top_k(entries):
    serialized = []
    for entry in entries or []:
        char = str((entry or {}).get("char") or "")
        conf = round(float((entry or {}).get("conf", 0.0)), 6)
        if not char:
            continue
        serialized.append({
            "char": char,
            "conf": conf,
        })
    return serialized


def _build_model_top_k(roi_images, top_k=5):
    if not roi_images:
        return []

    try:
        return [
            _serialize_top_k(item)
            for item in predict_characters_top_k(roi_images, normalized=True, top_k=top_k)
        ]
    except Exception:
        return [[] for _ in roi_images]


def _analysis_dir(analysis_id):
    return ANALYSES_DIR / analysis_id


def _analysis_manifest_path(analysis_id):
    return _analysis_dir(analysis_id) / "analysis.json"


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_analysis_record(analysis_id):
    return _load_json(_analysis_manifest_path(analysis_id))


def _append_feedback_log(entries):
    if not entries:
        return

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_feedback_message(num_corrections, num_rejections):
    parts = []
    if num_corrections:
        noun = "correction" if num_corrections == 1 else "corrections"
        parts.append(f"{num_corrections} {noun}")
    if num_rejections:
        noun = "rejection" if num_rejections == 1 else "rejections"
        parts.append(f"{num_rejections} {noun}")
    if not parts:
        return "No feedback items were saved."
    return "Saved " + " and ".join(parts) + "."


def _save_feedback_image(source_path, destination_path):
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read saved ROI image: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination_path), image):
        raise OSError(f"Could not write feedback image to: {destination_path}")


@app.context_processor
def inject_asset_version():
    return {
        "asset_version": _asset_version,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Please choose or drop an image first."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "The uploaded file is not valid."}), 400

    file_path = UPLOAD_FOLDER / "temp_upload.jpg"
    file.save(str(file_path))

    input_mode = request.form.get("input_mode", "upload").strip().lower()
    if input_mode not in {"upload", "draw"}:
        input_mode = "upload"

    try:
        roi_images, rects, thresh, img_display = segment_image(
            str(file_path),
            input_mode=input_mode,
        )
        if not roi_images:
            return jsonify({"error": "No characters were detected in the image."}), 400

        clean_display = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if clean_display is not None:
            img_display = clean_display

        raw_predictions = build_raw_predictions(roi_images)
        model_top_k_predictions = _build_model_top_k(roi_images, top_k=5)
        for item, top_k in zip(raw_predictions, model_top_k_predictions):
            item["top_k"] = top_k

        line_predictions = refine_predictions_by_line(rects, roi_images, raw_predictions)
        if not line_predictions:
            return jsonify({
                "error": "The detected characters could not be grouped into a valid expression line."
            }), 400

        analysis_created_at = _utc_now()
        analysis_id = _new_analysis_id(analysis_created_at)
        analysis_directory = _analysis_dir(analysis_id)
        characters_directory = analysis_directory / "characters"
        characters_directory.mkdir(parents=True, exist_ok=True)

        flattened_predictions = []
        stored_characters = []
        lines_response = []
        character_index = 0

        for line_idx, line in enumerate(line_predictions):
            color = LINE_COLORS[line_idx % len(LINE_COLORS)]
            characters = line["characters"]
            raw_chars = [item["char"] for item in characters]
            expression_str, result_str, error = build_and_evaluate(raw_chars)

            line_character_summaries = []

            for item in characters:
                x, y, w, h = item["rect"]
                label = f"L{line_idx + 1}: {item['char']} ({item['conf']:.2f})"
                cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    img_display,
                    label,
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                character_id = f"char_{character_index:03d}"
                normalized_roi = _extract_character_roi(thresh, item["rect"])
                roi_path = characters_directory / f"{character_id}.png"
                cv2.imwrite(str(roi_path), normalized_roi)

                character_payload = {
                    **item,
                    "id": character_id,
                    "line_index": line_idx,
                    "rect": _serialize_rect(item["rect"]),
                    "roi_image": _encode_png_data_uri(normalized_roi),
                }
                flattened_predictions.append(character_payload)
                stored_characters.append({
                    **item,
                    "id": character_id,
                    "line_index": line_idx,
                    "rect": _serialize_rect(item["rect"]),
                    "roi_path": str(roi_path),
                })
                line_character_summaries.append({
                    **item,
                    "rect": _serialize_rect(item["rect"]),
                })
                character_index += 1

            lx, ly, lw, lh = line["rect"]
            cv2.rectangle(img_display, (lx, ly), (lx + lw, ly + lh), color, 3)
            cv2.putText(
                img_display,
                f"Line {line_idx + 1}",
                (lx, max(30, ly - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            lines_response.append({
                "line_index": line_idx,
                "expression": expression_str,
                "result": result_str,
                "error": error,
                "characters": line_character_summaries,
                "rect": _serialize_rect(line["rect"]),
            })

        if len(lines_response) == 1:
            expression_str = lines_response[0]["expression"]
            result_str = lines_response[0]["result"]
            error = lines_response[0]["error"]
        else:
            success_count = sum(1 for item in lines_response if not item["error"])
            expression_str = f"Detected {len(lines_response)} lines"
            result_str = f"{success_count}/{len(lines_response)} lines evaluated successfully"
            error = None

        _save_json(_analysis_manifest_path(analysis_id), {
            "analysis_id": analysis_id,
            "created_at": _utc_iso(analysis_created_at),
            "input_mode": input_mode,
            "source_filename": file.filename,
            "expression": expression_str,
            "result": result_str,
            "error": error,
            "characters": stored_characters,
        })

        thresh_b64 = _encode_png_data_uri(thresh)
        display_b64 = _encode_png_data_uri(img_display)

        return jsonify({
            "analysis_id": analysis_id,
            "expression": expression_str,
            "result": result_str,
            "error": error,
            "input_mode": input_mode,
            "characters": flattened_predictions,
            "lines": lines_response,
            "display_image": display_b64,
            "thresh_image": thresh_b64,
        })

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"A server error occurred: {exc}"}), 500


@app.route("/api/feedback", methods=["POST"])
def save_feedback():
    payload = request.get_json(silent=True) or {}
    analysis_id = str(payload.get("analysis_id") or "").strip()
    corrections = payload.get("corrections") or []
    rejections = payload.get("rejections") or []

    if not analysis_id:
        return jsonify({"error": "Missing analysis_id."}), 400
    if not isinstance(corrections, list) or not isinstance(rejections, list):
        return jsonify({"error": "Corrections and rejections must be arrays."}), 400

    analysis_record = _load_analysis_record(analysis_id)
    if analysis_record is None:
        return jsonify({"error": "The selected analysis could not be found."}), 404

    character_lookup = {
        str(item["id"]): item
        for item in analysis_record.get("characters", [])
        if item.get("id")
    }

    normalized_corrections = {}
    for item in corrections:
        character_id = str((item or {}).get("character_id") or "").strip()
        corrected_char = _normalize_feedback_label((item or {}).get("corrected_char"))
        if not character_id:
            return jsonify({"error": "A correction is missing character_id."}), 400
        if corrected_char not in CORRECTABLE_LABELS:
            return jsonify({"error": f"Unsupported correction label: {corrected_char or '(empty)'}"}), 400
        if character_id not in character_lookup:
            return jsonify({"error": f"Unknown character_id: {character_id}"}), 400
        normalized_corrections[character_id] = corrected_char

    normalized_rejections = []
    for item in rejections:
        character_id = str((item or {}).get("character_id") or "").strip()
        if not character_id:
            return jsonify({"error": "A rejection is missing character_id."}), 400
        if character_id not in character_lookup:
            return jsonify({"error": f"Unknown character_id: {character_id}"}), 400
        normalized_rejections.append(character_id)

    overlap = set(normalized_corrections).intersection(normalized_rejections)
    if overlap:
        return jsonify({"error": "A character cannot be corrected and rejected at the same time."}), 400

    timestamp = _utc_now()
    saved_corrections = []
    saved_rejections = []

    try:
        for character_id, corrected_char in normalized_corrections.items():
            character = character_lookup[character_id]
            predicted_token = _safe_label_token(character.get("char", ""))
            label_token = _safe_label_token(corrected_char)
            output_dir = CORRECTIONS_DIR / label_token
            output_path = output_dir / (
                f"{analysis_id}_{character_id}_pred-{predicted_token}_label-{label_token}.png"
            )
            _save_feedback_image(character["roi_path"], output_path)

            saved_corrections.append({
                "saved_at": _utc_iso(timestamp),
                "analysis_id": analysis_id,
                "source_filename": analysis_record.get("source_filename", ""),
                "character_id": character_id,
                "predicted_char": character.get("char", ""),
                "corrected_char": corrected_char,
                "confidence": round(float(character.get("conf", 0.0)), 3),
                "raw_char": character.get("raw_char", character.get("char", "")),
                "raw_conf": round(float(character.get("raw_conf", character.get("conf", 0.0))), 3),
                "adjusted": bool(character.get("adjusted", False)),
                "line_index": int(character.get("line_index", 0)),
                "rect": _serialize_rect(character.get("rect", [0, 0, 0, 0])),
                "saved_image_path": os.path.relpath(output_path, FEEDBACK_ROOT),
                "action": "corrected",
            })

        for character_id in normalized_rejections:
            character = character_lookup[character_id]
            predicted_token = _safe_label_token(character.get("char", ""))
            output_dir = REJECTIONS_DIR / f"pred_{predicted_token}"
            output_path = output_dir / (
                f"{analysis_id}_{character_id}_pred-{predicted_token}_rejected.png"
            )
            _save_feedback_image(character["roi_path"], output_path)

            saved_rejections.append({
                "saved_at": _utc_iso(timestamp),
                "analysis_id": analysis_id,
                "source_filename": analysis_record.get("source_filename", ""),
                "character_id": character_id,
                "predicted_char": character.get("char", ""),
                "corrected_char": None,
                "confidence": round(float(character.get("conf", 0.0)), 3),
                "raw_char": character.get("raw_char", character.get("char", "")),
                "raw_conf": round(float(character.get("raw_conf", character.get("conf", 0.0))), 3),
                "adjusted": bool(character.get("adjusted", False)),
                "line_index": int(character.get("line_index", 0)),
                "rect": _serialize_rect(character.get("rect", [0, 0, 0, 0])),
                "saved_image_path": os.path.relpath(output_path, FEEDBACK_ROOT),
                "action": "rejected",
            })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500

    entries = [*saved_corrections, *saved_rejections]
    _append_feedback_log(entries)

    report_timestamp = _utc_now()
    report_path = REPORTS_DIR / f"{analysis_id}_{report_timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    _save_json(report_path, {
        "analysis_id": analysis_id,
        "created_at": _utc_iso(report_timestamp),
        "source_filename": analysis_record.get("source_filename", ""),
        "expression": analysis_record.get("expression", ""),
        "result": analysis_record.get("result", ""),
        "error": analysis_record.get("error"),
        "note": "",
        "corrections": saved_corrections,
        "rejections": saved_rejections,
    })

    return jsonify({
        "message": _build_feedback_message(len(saved_corrections), len(saved_rejections)),
        "saved_corrections": len(saved_corrections),
        "saved_rejections": len(saved_rejections),
        "report_path": os.path.relpath(report_path, PROJECT_ROOT),
    })


if __name__ == "__main__":
    print("Web server running at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
