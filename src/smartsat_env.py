import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from cdcl_baseline import SATInstance, CDCLSolver

def extract_sat_features(filepath: str, n_features: int = 48) -> np.ndarray:
    try:
        from satfeatpy import SATInstance as SFInst
        sf = SFInst(filepath)
        feats = sf.get_features()
        arr = np.array(list(feats.values()), dtype=np.float32)
        # Đảm bảo đúng 48 features
        if len(arr) >= n_features:
            arr = arr[:n_features]
        else:
            arr = np.pad(arr, (0, n_features - len(arr)))
        # Normalize (clip + scale)
        arr = np.clip(arr, -1e6, 1e6)
        arr = arr / (np.abs(arr).max() + 1e-8)
        return arr.astype(np.float32)
    except Exception:
        return np.zeros(n_features, dtype=np.float32)

N_VARS    = 20
N_CLAUSES = 91
N_GLOBAL  = 48

OBS_SIZE = N_VARS + N_CLAUSES + N_VARS * N_CLAUSES + N_GLOBAL
# 20 + 91 + 1820 + 48 = 1979


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
        self._global_features = np.zeros(N_GLOBAL, dtype=np.float32)

    # ---- Helpers ----

    def _load_instance(self, filepath: str):
        self._filepath = filepath
        inst = SATInstance.from_dimacs(filepath)
        self._solver = CDCLSolver(inst)
        self._global_features = extract_sat_features(filepath, N_GLOBAL)

    def _get_obs(self) -> np.ndarray:
        solver = self._solver
        n = N_VARS

        # Variable assignments
        var_assign = np.array(
            [solver.assignment[v] for v in range(1, n + 1)],
            dtype=np.float32
        )  # shape (20,)

        # Clause evaluations
        clause_eval = np.zeros(N_CLAUSES, dtype=np.float32)
        for ci, clause in enumerate(solver.clauses[:N_CLAUSES]):
            vals = [solver._lit_value(lit) for lit in clause]
            if 1 in vals:
                clause_eval[ci] = 1.0
            elif all(v == -1 for v in vals):
                clause_eval[ci] = -1.0
            else:
                clause_eval[ci] = 0.0

        # Bipartite graph (clause × var)
        graph = np.zeros((N_CLAUSES, n), dtype=np.float32)
        for ci, clause in enumerate(solver.clauses[:N_CLAUSES]):
            for lit in clause:
                vi = abs(lit) - 1
                if 0 <= vi < n:
                    graph[ci, vi] = 1.0

        obs = np.concatenate([
            var_assign,
            clause_eval,
            graph.flatten(),
            self._global_features
        ])
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        reward = 0.0
        for clause in self._solver.clauses[:N_CLAUSES]:
            vals = [self._solver._lit_value(lit) for lit in clause]
            if 1 in vals:
                reward += 1.0
            elif all(v == -1 for v in vals):
                reward -= 1.0
        return reward

    # ---- Gym Interface ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Chọn instance theo vòng (hoặc random)
        filepath = self.cnf_files[self.current_file_idx % len(self.cnf_files)]
        self.current_file_idx += 1
        self._load_instance(filepath)
        self._done = False

        # Unit propagation ban đầu (level 0)
        self._solver.unit_propagate()

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

        # Nếu biến đã gán → chọn biến chưa gán đầu tiên thay thế
        if solver.assignment[var] != 0:
            unassigned = [v for v in range(1, N_VARS + 1)
                          if solver.assignment[v] == 0]
            if not unassigned:
                # Tất cả biến đã gán → kết thúc
                sat = self._check_sat()
                reward = self._compute_reward()
                self._done = True
                return self._get_obs(), reward, True, False, {"sat": sat}
            var = unassigned[0]

        # Apply assignment
        solver.current_level += 1
        solver.trail_lim.append(len(solver.trail))
        solver._assign(var, value, solver.current_level, antecedent=None)

        # BCP
        conflict_ci = solver.unit_propagate()

        terminated = False
        truncated  = False

        if conflict_ci is not None:
            if solver.current_level == 0:
                # UNSAT
                reward = self._compute_reward()
                self._done = True
                return self._get_obs(), reward, True, False, {"sat": False}

            learned, btlevel = solver.analyze_conflict(conflict_ci)
            solver.backtrack(btlevel)
            solver.clauses.append(learned)

            # Force unit nếu learned clause đơn
            if len(learned) == 1:
                unit_lit = learned[0]
                v2 = abs(unit_lit)
                val2 = 1 if unit_lit > 0 else -1
                if solver.assignment[v2] == 0:
                    solver._assign(v2, val2, solver.current_level,
                                   len(solver.clauses) - 1)
                    conflict_ci2 = solver.unit_propagate()
                    if conflict_ci2 is not None:
                        self._done = True
                        return self._get_obs(), self._compute_reward(), True, False, {"sat": False}

        # Kiểm tra SAT
        sat = self._check_sat()
        if sat:
            reward = self._compute_reward()
            self._done = True
            return self._get_obs(), reward, True, False, {"sat": True}

        reward = self._compute_reward()
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
