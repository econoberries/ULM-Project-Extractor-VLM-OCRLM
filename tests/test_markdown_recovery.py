"""Offline recovery contracts: preserve real OCR text without inventing structure."""
import unittest

from ulm_extractor.markdown_recovery import (
    FORMAT, ocr_lines, resolve_markdown,
)


def item(text, coords):
    return {"text": text, "bbox": coords, "source": "flat-ocr"}


class MarkdownRecoveryTests(unittest.TestCase):
    def test_recognized_text_replaces_image_placeholder_in_place(self):
        markdown = ('# Nutrition\\n\\n<div style="text-align: center;">'
                    '<img src="imgs/box.jpg" alt="Image" /></div>\\n\\n## Ingredients')
        pp = {"res": {
            "overall_ocr_res": {"rec_texts": [], "rec_boxes": []},
            "parsing_res_list": [
                {"block_label": "image", "block_bbox": [10, 30, 300, 200],
                 "block_content": '<img src="imgs/box.jpg" />', "block_order": 0}
            ]
        }}
        lines = [item("Protein 8g", [40, 80, 220, 100]),
                 item("Calories 150", [40, 110, 220, 130])]
        resolved, audit = resolve_markdown(markdown, pp, lines)
        self.assertNotIn("<img", resolved)
        self.assertTrue(resolved.index("# Nutrition") <
                        resolved.index("Protein 8g") <
                        resolved.index("Calories 150") <
                        resolved.index("## Ingredients"))
        self.assertEqual(audit["text_lines_from_image_regions"], 2)
        self.assertEqual(audit["additional_ocr_lines"], 0)

    def test_missing_text_without_image_placeholder_is_not_dropped(self):
        pp = {"res": {
            "overall_ocr_res": {"rec_texts": [], "rec_boxes": []},
            "parsing_res_list": [],
        }}
        md, stats = resolve_markdown(
            "# Nutrition Facts",
            pp,
            [item("Calories 250", [5, 40, 100, 55]),
             item("Protein 4g", [5, 60, 100, 75])])
        self.assertTrue(md.startswith("# Nutrition Facts"))
        self.assertIn("Calories 250", md)
        self.assertIn("Protein 4g", md)
        self.assertEqual(stats["additional_ocr_lines"], 2)
        self.assertNotIn("| Calories |", md)  # no fabricated table cells

    def test_pp_table_remains_intact_and_existing_ocr_is_not_duplicated(self):
        table = "<table><tr><td>Energy</td><td>743 kJ</td></tr></table>"
        pp = {"res": {"overall_ocr_res": {"rec_texts": [], "rec_boxes": []},
                      "parsing_res_list": []}}
        content, stats = resolve_markdown(table, pp, [
            item("Energy", [10, 10, 40, 20]),
            item("743 kJ", [50, 10, 90, 20])])
        self.assertIn(table, content)
        self.assertEqual(stats["additional_ocr_lines"], 0)
        self.assertNotIn("Additional OCR", content)

    def test_markdown_crop_ocr_replaces_image_without_bbox(self):
        pp = {"res": {"overall_ocr_res": {"rec_texts": [], "rec_boxes": []},
                      "parsing_res_list": []}}
        img = '<img src="imgs/region.jpg" alt="Image" />'
        text, audit = resolve_markdown(
            "Header\\n\\n" + img + "\\n\\nFooter",
            pp, [], {"imgs/region.jpg": [item("Serving size 42 g", None)]})
        self.assertNotIn("<img", text)
        self.assertIn("Serving size 42 g", text)
        self.assertTrue(text.index("Header") < text.index("Serving size") <
                        text.index("Footer"))
        self.assertEqual(audit["text_lines_from_image_crops"], 1)

    def test_pp_internal_ocr_can_recover_text_not_in_flat_ocr(self):
        pp = {"res": {"overall_ocr_res": {
            "rec_texts": ["Sodium 144 mg"], "rec_polys": [
                [[10, 30], [80, 30], [80, 43], [10, 43]]
            ]}, "parsing_res_list": []}}
        result, audit = resolve_markdown("# Nutrition Facts", pp, [])
        self.assertIn("Sodium 144 mg", result)
        self.assertEqual(audit["pp_internal_ocr_lines"], 1)

    def test_parser_recognition_preserves_geometry(self):
        src = {"res": {"rec_texts": ["Sodium", "144mg"],
                       "rec_boxes": [[10, 20, 50, 40], [100, 20, 150, 40]]}}
        result = ocr_lines(src)
        self.assertEqual(result[1]["bbox"], [100.0, 20.0, 150.0, 40.0])
        self.assertIn("recovered", FORMAT)


if __name__ == "__main__":
    unittest.main()
