import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "LangSAT_Kaggle_Reproduce.ipynb"


class KaggleNotebookConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
        )

    def test_notebook_has_full_and_smoke_run_modes(self):
        self.assertIn("RUN_MODE", self.source)
        self.assertIn("smoke", self.source)
        self.assertIn("full", self.source)

    def test_train_cell_uses_configured_total_steps(self):
        self.assertNotIn('os.environ["LANGSAT_TOTAL_STEPS"] = "100000"\n\nfrom training_pipeline', self.source)

    def test_notebook_runs_repo_smoke_tests_on_kaggle(self):
        self.assertIn("unittest discover", self.source)


if __name__ == "__main__":
    unittest.main()
