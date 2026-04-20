import csv
import unittest
from pathlib import Path

from experiments.method_comparison.run_full_pipeline import RESULT_HEADERS
from lab.parsers import build_method_comparison_summary


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


class MethodComparisonResultSchemaTests(unittest.TestCase):
    def _read_header(self, csv_path):
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            return next(csv.reader(fh))

    def test_checked_in_full_pipeline_csv_headers_match_runtime_schema(self):
        csv_paths = sorted(RESULTS_DIR.glob("*_full_pipeline_results.csv"))
        self.assertTrue(csv_paths, "Expected checked-in full-pipeline CSV files.")

        for csv_path in csv_paths:
            with self.subTest(csv_path=csv_path.name):
                self.assertEqual(self._read_header(csv_path), RESULT_HEADERS)

    def test_checked_in_cora_results_are_still_parseable_by_django_summary(self):
        summary = build_method_comparison_summary(RESULTS_DIR / "cora_full_pipeline_results.csv")
        self.assertIn("methods", summary)
        self.assertGreaterEqual(len(summary["methods"]), 1)


if __name__ == "__main__":
    unittest.main()
