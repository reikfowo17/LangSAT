import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def write_cnf(path: Path, n_vars: int, clauses: list[list[int]]) -> None:
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    lines.extend(" ".join(str(lit) for lit in clause) + " 0" for clause in clauses)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CompetitiveAdvisorTests(unittest.TestCase):
    def test_jeroslow_wang_order_prefers_short_clause_weight(self):
        from src.cdcl_baseline import SATInstance
        from src.competitive_advisor import build_jeroslow_wang_order

        inst = SATInstance(
            4,
            [
                [1, 2, 3],
                [-1, 2, 3],
                [4],
            ],
        )

        order = build_jeroslow_wang_order(inst)

        self.assertEqual(order[0], (4, 1))

    def test_advisor_falls_back_when_rl_action_is_invalid(self):
        from src.cdcl_baseline import CDCLSolver, SATInstance
        from src.competitive_advisor import CompetitiveBranchingAdvisor

        class InvalidModel:
            def predict(self, obs, deterministic=True):
                return 999, None

        inst = SATInstance(3, [[1, 2], [1, -2], [3]])
        solver = CDCLSolver(inst)
        advisor = CompetitiveBranchingAdvisor(
            inst,
            rl_model=InvalidModel(),
            global_features=[0.0] * 48,
            enable_rl=True,
        )

        decision = advisor.decide(solver)

        self.assertEqual(decision, (3, 1))
        self.assertEqual(advisor.stats["rl_invalid"], 1)
        self.assertEqual(advisor.stats["static"], 1)

    def test_evaluate_accepts_competitive_policy_mode(self):
        original = os.environ.get("LANGSAT_POLICY_MODE")
        os.environ["LANGSAT_POLICY_MODE"] = "competitive"
        try:
            from src import evaluate

            reloaded = importlib.reload(evaluate)

            self.assertEqual(reloaded.SMARTSAT_POLICY_MODE, "competitive")
            self.assertEqual(reloaded.RUN_PROFILE, "competitive")
        finally:
            if original is None:
                os.environ.pop("LANGSAT_POLICY_MODE", None)
            else:
                os.environ["LANGSAT_POLICY_MODE"] = original

    def test_competitive_static_mode_does_not_require_model_file(self):
        original_mode = os.environ.get("LANGSAT_POLICY_MODE")
        original_enable_rl = os.environ.get("LANGSAT_COMPETITIVE_ENABLE_RL")
        os.environ["LANGSAT_POLICY_MODE"] = "competitive"
        os.environ["LANGSAT_COMPETITIVE_ENABLE_RL"] = "0"
        try:
            from src import evaluate

            reloaded = importlib.reload(evaluate)

            self.assertFalse(reloaded.model_required_for_policy())
        finally:
            if original_mode is None:
                os.environ.pop("LANGSAT_POLICY_MODE", None)
            else:
                os.environ["LANGSAT_POLICY_MODE"] = original_mode
            if original_enable_rl is None:
                os.environ.pop("LANGSAT_COMPETITIVE_ENABLE_RL", None)
            else:
                os.environ["LANGSAT_COMPETITIVE_ENABLE_RL"] = original_enable_rl

    def test_competitive_static_solve_does_not_need_satfeatpy(self):
        original_mode = os.environ.get("LANGSAT_POLICY_MODE")
        original_enable_rl = os.environ.get("LANGSAT_COMPETITIVE_ENABLE_RL")
        os.environ["LANGSAT_POLICY_MODE"] = "competitive"
        os.environ["LANGSAT_COMPETITIVE_ENABLE_RL"] = "0"
        try:
            from src import evaluate

            reloaded = importlib.reload(evaluate)
            clauses = [[1, 2, 3] for _ in range(91)]
            with tempfile.TemporaryDirectory() as tmp:
                cnf = Path(tmp) / "uf20.cnf"
                write_cnf(cnf, 20, clauses)
                with mock.patch.object(
                    reloaded,
                    "extract_sat_features",
                    side_effect=RuntimeError("SATfeatPy should not be called"),
                ):
                    sat, _, stats = reloaded.solve_with_smartsat(str(cnf), model=None)

            self.assertTrue(sat)
            self.assertGreater(stats["advisor_static_decisions"], 0)
        finally:
            if original_mode is None:
                os.environ.pop("LANGSAT_POLICY_MODE", None)
            else:
                os.environ["LANGSAT_POLICY_MODE"] = original_mode
            if original_enable_rl is None:
                os.environ.pop("LANGSAT_COMPETITIVE_ENABLE_RL", None)
            else:
                os.environ["LANGSAT_COMPETITIVE_ENABLE_RL"] = original_enable_rl


if __name__ == "__main__":
    unittest.main()
