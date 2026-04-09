import os
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segmentation.runtime_pipeline import analyze_expression_image


EXPECTED_EXPRESSIONS = {
    "2c2decf1-9444-4de6-8458-3c3c3a0ed250.jpg": "4+5-6/7*3+(5-4)",
    "2d4a10e7-2655-45ae-84ea-04c879264c31.jpg": "(6+3)/5*3*3",
    "40bb37dd-2d22-40f9-9620-13499c44113a.jpg": "7/3+(5-6+2)*2",
    "72f12e43-dafb-437e-a6ec-57a2ef48fafa (1).jpg": "456910332",
    "79e6b45c-225f-4bd7-887e-f742eead6c64.jpg": "176543059",
    "b90f1184-d1a2-4a5b-b44f-4b1baf54cb6b.jpg": "1750996",
    "bb7a1eab-d358-4b18-a0d6-c92f186d17d3.jpg": "(1*7)*3+5/6",
    "c7f08e1e-2aa0-4766-8feb-39bdc8b5b545.jpg": "(5+3-2)*8/3",
    "image.png": "774913506",
    "test.jpg": "456910332",
    "Unatitled.png": "4+5-6/7*3+(5-4)",
    "Untitled.jpg": "0906688341",
    "Untitled.png": "754321007",
    "Untitled1.jpg": "176544322",
}


class ImgFolderRegressionTests(unittest.TestCase):
    def test_img_folder_cases_match_expected_expression(self):
        img_dir = ROOT_DIR / "img"
        self.assertTrue(img_dir.exists(), f"Missing img directory: {img_dir}")

        for file_name, expected_expression in EXPECTED_EXPRESSIONS.items():
            image_path = img_dir / file_name
            with self.subTest(image=file_name):
                self.assertTrue(image_path.exists(), f"Missing image: {image_path}")

                analysis = analyze_expression_image(
                    str(image_path),
                    input_mode="upload",
                    include_model_top_k=True,
                    top_k=5,
                )

                self.assertIsNone(
                    analysis["fatal_error"],
                    f"Unexpected pipeline failure for {file_name}: {analysis['fatal_error']}",
                )
                self.assertEqual(expected_expression, analysis["expression"])

                line_characters = [
                    item["char"]
                    for line in analysis["lines"]
                    for item in line["characters"]
                ]
                self.assertEqual(len(expected_expression), len(line_characters))
                self.assertEqual(expected_expression, "".join(line_characters))


if __name__ == "__main__":
    unittest.main()
