import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# Đảm bảo import được từ cùng thư mục src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdcl_baseline import SATInstance, CDCLSolver
from satfeat_adapter import extract_sat_features


_BASELINE_DECISION_CACHE: dict[str, int] = {}


N_VARS    = 20
N_CLAUSES = 91
N_GLOBAL  = 48
REWARD_MODE = os.environ.get("LANGSAT_REWARD_MODE", "paper").lower()

OBS_SIZE = N_VARS + N_CLAUSES + N_VARS * N_CLAUSES + N_GLOBAL
# 20 + 91 + 1820 + 48 = 1979


def build_solver_observation(
    solver: CDCLSolver,
    global_features: np.ndarray,
) -> np.ndarray:
    n = N_VARS

    var_assign = np.array(
        [
            solver.assignment[v] if v <= solver.inst.n_vars else 0
            for v in range(1, n + 1)
        ],
        dtype=np.float32,
    )

    clause_eval = np.zeros(N_CLAUSES, dtype=np.float32)
    for ci, clause in enumerate(solver.clauses[:N_CLAUSES]):
        vals = [solver._lit_value(lit) for lit in clause]
        if 1 in vals:
            clause_eval[ci] = 1.0
        elif all(v == -1 for v in vals):
            clause_eval[ci] = -1.0

    graph = np.zeros((N_CLAUSES, n), dtype=np.float32)
    for ci, clause in enumerate(solver.clauses[:N_CLAUSES]):
        for lit in clause:
            vi = abs(lit) - 1
            if 0 <= vi < n:
                graph[ci, vi] = 1.0 if lit > 0 else -1.0

    if len(global_features) >= N_GLOBAL:
        features = global_features[:N_GLOBAL]
    else:
        features = np.pad(global_features, (0, N_GLOBAL - len(global_features)))

    obs = np.concatenate([var_assign, clause_eval, graph.flatten(), features])
    return obs.astype(np.float32)


class SmartSATEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cnf_files: list[str], render_mode=None):
        super().__init__()
        self.cnf_files = cnf_files
        self.current_file_idx = 0
        self._rng = np.random.default_rng(42)

        # Observation: flat vector
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

        # Action: 40 = 20 biến × 2 giá trị
        self.action_space = spaces.Discrete(N_VARS * 2)

        # State
        self._solver: Optional[CDCLSolver] = None
        self._filepath: Optional[str] = None
        self._done = False
        self._step_count = 0
        self._max_steps = N_VARS * 2   # Tối đa 40 steps, tránh episode chạy vô hạn
        self._global_features = np.zeros(N_GLOBAL, dtype=np.float32)
        self.preferred_literals: list[int] = []
        self._baseline_decisions = N_VARS

    # ---- Helpers ----

    def _load_instance(self, filepath: str):
        self._filepath = filepath
        inst = SATInstance.from_dimacs(filepath)
        self._solver = CDCLSolver(inst)
        self._global_features = extract_sat_features(filepath, N_GLOBAL)
        self._baseline_decisions = (
            baseline_decisions(filepath) if REWARD_MODE == "shaped" else N_VARS
        )

    def _get_obs(self) -> np.ndarray:
        return build_solver_observation(self._solver, self._global_features)

    def _compute_reward(self) -> float:
        satisfied = 0
        unsatisfied = 0
        for clause in self._solver.clauses[:N_CLAUSES]:
            vals = [self._solver._lit_value(lit) for lit in clause]
            if 1 in vals:
                satisfied += 1
            elif all(v == -1 for v in vals):
                unsatisfied += 1
        if REWARD_MODE == "paper":
            return float(satisfied - unsatisfied)
        return (satisfied / N_CLAUSES) - float(unsatisfied)

    def _terminal_reward(self, sat: bool) -> float:
        if REWARD_MODE == "paper":
            return self._compute_reward() if sat else -float(N_CLAUSES + self._step_count)
        if not sat:
            return -20.0 - self._step_count
        baseline = max(self._baseline_decisions, 1)
        decision_delta = baseline - self._solver.stats.decisions
        return 20.0 + (2.0 * decision_delta) - (0.5 * self._solver.stats.conflicts)

    # ---- Gym Interface ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Chọn instance theo vòng (hoặc random)
        filepath = self.cnf_files[self.current_file_idx % len(self.cnf_files)]
        self.current_file_idx += 1
        self._load_instance(filepath)
        self._done = False
        self._step_count = 0
        self.preferred_literals = []

        # Unit propagation ban đầu (level 0)
        self._solver._find_initial_units()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        if self._done:
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        solver = self._solver
        var_idx = action // 2
        value   = 1 if action % 2 == 1 else -1
        var     = var_idx + 1

        # Match evaluation: invalid policy actions fall back to the baseline
        # branching heuristic instead of silently choosing the first variable.
        if solver.assignment[var] != 0:
            fallback = solver.pick_branching_variable()
            if fallback is None:
                # Tất cả biến đã gán → kết thúc
                sat = self._check_sat()
                reward = self._terminal_reward(sat)
                self._done = True
                return self._get_obs(), reward, True, False, {"sat": sat}
            var, value = fallback

        lit = var if value == 1 else -var
        if lit not in self.preferred_literals and -lit not in self.preferred_literals:
            self.preferred_literals.append(lit)

        # Apply assignment
        solver.current_level += 1
        solver.trail_lim.append(len(solver.trail))
        solver._enqueue(var, value, solver.current_level, reason=None)

        # BCP
        conflict_ci = solver.unit_propagate()

        self._step_count += 1
        terminated = False
        truncated  = False

        if conflict_ci is not None:
            if solver.current_level == 0:
                # UNSAT
                self._done = True
                return self._get_obs(), self._terminal_reward(False), True, False, {"sat": False}

            learned, btlevel = solver.analyze_conflict(conflict_ci)
            solver.backtrack(btlevel)
            ci_new = len(solver.clauses)
            solver.clauses.append(learned)

            # Force unit nếu learned clause đơn
            if len(learned) == 1:
                unit_lit = learned[0]
                v2 = abs(unit_lit)
                val2 = 1 if unit_lit > 0 else -1
                if solver.assignment[v2] == 0:
                    solver._enqueue(v2, val2, solver.current_level, ci_new)
                    conflict_ci2 = solver.unit_propagate()
                    if conflict_ci2 is not None:
                        self._done = True
                        return self._get_obs(), self._terminal_reward(False), True, False, {"sat": False}

        # Kiểm tra SAT
        sat = self._check_sat()
        if sat:
            self._done = True
            return self._get_obs(), self._terminal_reward(True), True, False, {"sat": True}

        reward = self._compute_reward() - 0.05 - (0.25 if conflict_ci is not None else 0.0)

        # Truncation: episode quá dài → dừng lại
        if self._step_count >= self._max_steps:
            return self._get_obs(), reward - 5.0, False, True, {
                "truncated": True,
                "preferred_literals": list(self.preferred_literals),
            }

        return self._get_obs(), reward, False, False, {}

    def _check_sat(self) -> bool:
        solver = self._solver
        for clause in solver.clauses[:N_CLAUSES]:
            vals = [solver._lit_value(lit) for lit in clause]
            if 1 not in vals:
                return False
        return True

    def render(self):
        pass

    def close(self):
        pass


def baseline_decisions(filepath: str) -> int:
    cached = _BASELINE_DECISION_CACHE.get(filepath)
    if cached is not None:
        return cached
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    solver.solve()
    decisions = max(solver.stats.decisions, 1)
    _BASELINE_DECISION_CACHE[filepath] = decisions
    return decisions
