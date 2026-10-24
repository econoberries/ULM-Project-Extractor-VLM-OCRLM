"""Check GPU smoke-test scripts without importing CUDA frameworks or allocating a GPU."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class SmokeScriptContractTests(unittest.TestCase):
    def test_python_scripts_parse(self):
        import ast
        for path in sorted((ROOT / "scripts").glob("check_*.py")):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"))

    def test_paddleocr_result_parser_handles_dict_and_json(self):
        module_path = ROOT / "scripts" / "check_paddleocr.py"
        spec = importlib.util.spec_from_file_location("smoke_paddleocr", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.recognized_lines({"res": {"rec_texts": ["abc", "123"]}}),
                         ["abc", "123"])
        self.assertEqual(module.recognized_lines('{"res":{"rec_texts":["test"]}}'), ["test"])
        with self.assertRaises(ValueError):
            module.recognized_lines({"res": {"wrong": []}})

if __name__ == "__main__":
    unittest.main()
