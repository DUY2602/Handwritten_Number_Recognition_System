from .segmentation import segment_image


__all__ = [
    "segment_image",
    "build_corrected_predictions",
    "build_raw_predictions",
    "correct_sequence",
    "build_and_evaluate",
    "run_expression_pipeline",
    "predict_character",
    "refine_predictions",
    "refine_predictions_by_line",
]


def __getattr__(name):
    if name in {"build_corrected_predictions", "build_raw_predictions", "correct_sequence"}:
        from .context_corrector import build_corrected_predictions, build_raw_predictions, correct_sequence

        namespace = {
            "build_corrected_predictions": build_corrected_predictions,
            "build_raw_predictions": build_raw_predictions,
            "correct_sequence": correct_sequence, # context_corrector is still in src/segmentation
        }
        return namespace[name]
    if name == "build_and_evaluate":
        from .expression_parser import build_and_evaluate

        return build_and_evaluate
    if name == "run_expression_pipeline":
        from .main_extension import run_expression_pipeline

        return run_expression_pipeline
    if name == "predict_character": # operator_classifier moved to src/model
        from model.operator_classifier import predict_character

        return predict_character
    if name in {"refine_predictions", "refine_predictions_by_line"}:
        from .prediction_refiner import refine_predictions, refine_predictions_by_line

        namespace = {
            "refine_predictions": refine_predictions,
            "refine_predictions_by_line": refine_predictions_by_line,
        }
        return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
