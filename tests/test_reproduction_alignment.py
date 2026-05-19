import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_cnf(path: Path, n_vars: int, clauses: list[list[int]]) -> None:
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    lines.extend(" ".join(str(lit) for lit in clause) + " 0" for clause in clauses)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ReproductionAlignmentTests(unittest.TestCase):
    def test_paper_reward_is_bounded_by_clause_score(self):
        from src import smartsat_env

        clauses = [[1, 2, 3] for _ in range(45)]
        clauses.extend([[4, 5, 6] for _ in range(smartsat_env.N_CLAUSES - 45)])
        with tempfile.TemporaryDirectory() as tmp:
            cnf = Path(tmp) / "uf20_like.cnf"
            write_cnf(cnf, smartsat_env.N_VARS, clauses)

            with mock.patch.object(
                smartsat_env,
                "extract_sat_features",
                return_value=np.zeros(smartsat_env.N_GLOBAL, dtype=np.float32),
            ):
                env = smartsat_env.SmartSATEnv([str(cnf)])
                env.reset()
                _, first_reward, first_done, _, _ = env.step(1)
                _, second_reward, terminated, truncated, _ = env.step(7)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(first_done)
        self.assertEqual(first_reward + second_reward, smartsat_env.N_CLAUSES)

    def test_smartsat_rejects_non_uf20_91_instances(self):
        from src import smartsat_env

        with tempfile.TemporaryDirectory() as tmp:
            cnf = Path(tmp) / "small.cnf"
            write_cnf(cnf, 3, [[1, -2, 3]])

            with mock.patch.object(
                smartsat_env,
                "extract_sat_features",
                return_value=np.zeros(smartsat_env.N_GLOBAL, dtype=np.float32),
            ):
                env = smartsat_env.SmartSATEnv([str(cnf)])
                with self.assertRaisesRegex(ValueError, "uf20-91"):
                    env.reset()

    def test_evaluation_ignores_paper_time_scale_env(self):
        original = os.environ.get("LANGSAT_REPORT_SCALE_TO_PAPER")
        os.environ["LANGSAT_REPORT_SCALE_TO_PAPER"] = "1"
        try:
            from src import evaluate

            reloaded = importlib.reload(evaluate)

            self.assertEqual(reloaded._metric_basis(), "raw_total_time")
            self.assertFalse(hasattr(reloaded, "REPORT_SCALE_TO_PAPER"))
        finally:
            if original is None:
                os.environ.pop("LANGSAT_REPORT_SCALE_TO_PAPER", None)
            else:
                os.environ["LANGSAT_REPORT_SCALE_TO_PAPER"] = original

    def test_satfeatpy_defaults_to_full_local_search(self):
        os.environ.pop("LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH", None)
        from src import satfeat_adapter

        reloaded = importlib.reload(satfeat_adapter)

        self.assertTrue(reloaded.SATFEATPY_FULL_LOCAL_SEARCH)


if __name__ == "__main__":
    unittest.main()
