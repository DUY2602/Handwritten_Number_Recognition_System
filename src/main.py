import argparse

from segmentation.main_extension import run_expression_pipeline


def run_pipeline(image_path, show_visualization=True):
    return run_expression_pipeline(image_path, show_visualization=show_visualization)


def main():
    parser = argparse.ArgumentParser(description="Handwritten Math Expression Recognition System")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run the pipeline without showing the matplotlib result window.",
    )
    args = parser.parse_args()

    expression_str, result_str, error = run_pipeline(
        args.input,
        show_visualization=not args.no_display,
    )

    print("Recognized expression:", expression_str or "<empty>")
    if error:
        print("Error:", error)
        raise SystemExit(1)

    print("Result:", result_str)


if __name__ == "__main__":
    main()
