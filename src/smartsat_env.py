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


N_VARS    = 20
N_CLAUSES = 91
N_GLOBAL  = 48
MAX_STEPS = int(os.environ.get("LANGSAT_ENV_MAX_STEPS", str(N_VARS * 10)))

OBS_SIZE = N_VARS + N_CLAUSES + N_VARS * N_CLAUSES + N_GLOBAL
# 20 + 91 + 1820 + 48 = 1979
INVALID_ACTION_PENALTY = float(os.environ.get("LANGSAT_INVALID_ACTION_PENALTY", "2.0"))


def validate_uf20_91_instance(inst: SATInstance, filepath: str = ""):
    if inst.n_vars != N_VARS or inst.n_clauses != N_CLAUSES:
        location = f" for {filepath}" if filepath else ""
        raise ValueError(
            "SmartSAT paper reproduction is fixed to uf20-91 "
            f"({N_VARS} variables, {N_CLAUSES} clauses); got "
            f"{inst.n_vars} variables and {inst.n_clauses} clauses{location}."
        )


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
        self._max_steps = MAX_STEPS
        self._global_features = np.zeros(N_GLOBAL, dtype=np.float32)
        self.preferred_literals: list[int] = []
        self._invalid_actions = 0
        self._last_action_mask = np.ones(N_VARS * 2, dtype=np.int8)
        self._last_clause_score = 0.0

    # ---- Helpers ----

    def _load_instance(self, filepath: str):
        self._filepath = filepath
        inst = SATInstance.from_dimacs(filepath)
        validate_uf20_91_instance(inst, filepath)
        self._solver = CDCLSolver(inst)
        self._global_features = extract_sat_features(filepath, N_GLOBAL)

    def _get_obs(self) -> np.ndarray:
        return build_solver_observation(self._solver, self._global_features)

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(N_VARS * 2, dtype=np.int8)
        solver = self._solver
        if solver is None:
            return mask
        for var in range(1, min(self._solver.inst.n_vars, N_VARS) + 1):
            if solver.assignment[var] == 0:
                mask[(var - 1) * 2] = 1
                mask[(var - 1) * 2 + 1] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        return self._last_action_mask.copy()

    def _clause_score(self) -> float:
        satisfied = 0
        unsatisfied = 0
        for clause in self._solver.clauses[:N_CLAUSES]:
            vals = [self._solver._lit_value(lit) for lit in clause]
            if 1 in vals:
                satisfied += 1
            elif all(v == -1 for v in vals):
                unsatisfied += 1
        return float(satisfied - unsatisfied)

    def _compute_reward(self) -> float:
        score = self._clause_score()
        reward = score - self._last_clause_score
        self._last_clause_score = score
        return reward

    def _terminal_reward(self, sat: bool) -> float:
        return self._compute_reward() if sat else -float(N_CLAUSES)

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
        self._invalid_actions = 0

        # Unit propagation ban đầu (level 0)
        self._solver._find_initial_units()
        self._last_clause_score = self._clause_score()
        self._last_action_mask = self._compute_action_mask()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        if self._done:
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        solver = self._solver
        self._last_action_mask = self._compute_action_mask()
        var_idx = action // 2
        value   = 1 if action % 2 == 1 else -1
        var     = var_idx + 1
        invalid_action = var_idx < 0 or var_idx >= N_VARS or solver.assignment[var] != 0

        if invalid_action:
            self._invalid_actions += 1
            self._done = True
            reward = -float(N_CLAUSES) - INVALID_ACTION_PENALTY
            return self._get_obs(), reward, True, False, self._info(invalid_action=True)

        lit = var if value == 1 else -var
        if lit not in self.preferred_literals and -lit not in self.preferred_literals:
            self.preferred_literals.append(lit)

        # Apply assignment
        solver.make_decision(var, value)

        # BCP
        conflict_ci = solver.unit_propagate()

        self._step_count += 1

        if conflict_ci is not None:
            if solver.current_level == 0:
                # UNSAT
                self._done = True
                return self._get_obs(), self._terminal_reward(False), True, False, self._info(sat=False)

            learned, _ = solver.learn_from_conflict(conflict_ci)

            # Force unit nếu learned clause đơn
            if len(learned) == 1:
                conflict_ci2 = solver.unit_propagate()
                if conflict_ci2 is not None:
                    self._done = True
                    return self._get_obs(), self._terminal_reward(False), True, False, self._info(sat=False)

        # Kiểm tra SAT
        sat = self._check_sat()
        if sat:
            self._done = True
            return self._get_obs(), self._terminal_reward(True), True, False, self._info(sat=True)

        reward = self._compute_reward()

        # Truncation: episode quá dài → dừng lại
        if self._step_count >= self._max_steps:
            return self._get_obs(), reward, False, True, self._info(truncated=True)

        self._last_action_mask = self._compute_action_mask()
        return self._get_obs(), reward, False, False, {}

    def _info(
        self,
        sat: Optional[bool] = None,
        truncated: bool = False,
        invalid_action: bool = False,
    ) -> dict:
        info = {
            "steps": self._step_count,
            "invalid_actions": self._invalid_actions,
            "preferred_literals": list(self.preferred_literals),
        }
        if sat is not None:
            info["sat"] = sat
        if truncated:
            info["truncated"] = True
        if invalid_action:
            info["invalid_action"] = True
        return info

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
