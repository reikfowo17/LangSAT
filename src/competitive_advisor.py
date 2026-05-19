from __future__ import annotations

import time
from collections import Counter
from typing import Iterable, Optional

import numpy as np

try:
    from .cdcl_baseline import CDCLSolver, SATInstance
    from .smartsat_env import N_GLOBAL, N_VARS, build_solver_observation
except ImportError:
    from cdcl_baseline import CDCLSolver, SATInstance
    from smartsat_env import N_GLOBAL, N_VARS, build_solver_observation


LiteralOrder = list[tuple[int, int]]


def build_jeroslow_wang_order(inst: SATInstance) -> LiteralOrder:
    """One-shot Jeroslow-Wang literal guidance for uf20-style SAT instances."""
    scores = {var: 0.0 for var in range(1, inst.n_vars + 1)}
    polarity = {var: 0.0 for var in range(1, inst.n_vars + 1)}
    occurrences = {var: 0 for var in range(1, inst.n_vars + 1)}
    shortest_clause = {var: 10**9 for var in range(1, inst.n_vars + 1)}

    for clause in inst.clauses:
        weight = 2.0 ** (-len(clause))
        for lit in clause:
            var = abs(lit)
            scores[var] += weight
            polarity[var] += weight if lit > 0 else -weight
            occurrences[var] += 1
            shortest_clause[var] = min(shortest_clause[var], len(clause))

    ranked = sorted(
        scores,
        key=lambda var: (scores[var], -shortest_clause[var], abs(polarity[var]), occurrences[var], -var),
        reverse=True,
    )
    return [(var, 1 if polarity[var] >= 0 else -1) for var in ranked]


def build_moms_order(inst: SATInstance) -> LiteralOrder:
    """Maximum Occurrences in clauses of Minimum Size, used as a fallback tie-breaker."""
    if not inst.clauses:
        return []

    min_len = min(len(clause) for clause in inst.clauses)
    pos = Counter()
    neg = Counter()

    for clause in inst.clauses:
        if len(clause) != min_len:
            continue
        for lit in clause:
            if lit > 0:
                pos[abs(lit)] += 1
            else:
                neg[abs(lit)] += 1

    def score(var: int) -> tuple[int, int, int]:
        p = pos[var]
        n = neg[var]
        return ((p + n) * 4 + p * n, p + n, -var)

    ranked = sorted(range(1, inst.n_vars + 1), key=score, reverse=True)
    return [(var, 1 if pos[var] >= neg[var] else -1) for var in ranked if pos[var] or neg[var]]


def build_occurrence_order(inst: SATInstance) -> LiteralOrder:
    """Fast occurrence guidance; strong for this repo's uf20-91 raw-time baseline."""
    pos = Counter()
    neg = Counter()
    unit_vars: set[int] = set()

    for clause in inst.clauses:
        if len(clause) == 1:
            unit_vars.add(abs(clause[0]))
        for lit in clause:
            if lit > 0:
                pos[abs(lit)] += 1
            else:
                neg[abs(lit)] += 1

    def polarity(var: int) -> int:
        return 1 if pos[var] >= neg[var] else -1

    def score(var: int) -> tuple[int, int, int, int]:
        total = pos[var] + neg[var]
        balance = abs(pos[var] - neg[var])
        return (1 if var in unit_vars else 0, total, balance, -var)

    ranked = sorted(range(1, inst.n_vars + 1), key=score, reverse=True)
    return [(var, polarity(var)) for var in ranked if pos[var] or neg[var] or var in unit_vars]


class CompetitiveBranchingAdvisor:
    """Fast branching advisor optimized for raw-time winrate against this repo's baseline.

    The default path is one-shot static guidance, inspired by recent SAT guidance
    work where expensive neural inference is kept out of the hot CDCL loop. An
    optional RL model can be tried first, but invalid or unavailable decisions
    fall back instead of aborting the solve.
    """

    def __init__(
        self,
        inst: SATInstance,
        rl_model=None,
        global_features: Optional[Iterable[float]] = None,
        enable_rl: bool = False,
    ):
        self.inst = inst
        self.rl_model = rl_model
        self.enable_rl = enable_rl and rl_model is not None
        self.global_features = _global_feature_array(global_features)
        self.static_order = _merge_orders(
            build_occurrence_order(inst),
            build_jeroslow_wang_order(inst),
            build_moms_order(inst),
            inst.n_vars,
        )
        self.stats = {
            "rl": 0,
            "rl_invalid": 0,
            "rl_time_raw": 0.0,
            "static": 0,
            "baseline": 0,
            "no_decision": 0,
        }

    def decide(self, solver: CDCLSolver) -> Optional[tuple[int, int]]:
        if self.enable_rl:
            decision = self._try_rl_decision(solver)
            if decision is not None:
                self.stats["rl"] += 1
                return decision

        static_decision = self._static_decision(solver)
        if static_decision is not None:
            self.stats["static"] += 1
            return static_decision

        fallback = solver.pick_branching_variable()
        if fallback is not None:
            self.stats["baseline"] += 1
            return fallback

        self.stats["no_decision"] += 1
        return None

    def _try_rl_decision(self, solver: CDCLSolver) -> Optional[tuple[int, int]]:
        start = time.perf_counter()
        try:
            obs = build_solver_observation(solver, self.global_features)
            action, _ = self.rl_model.predict(obs, deterministic=True)
            action = int(action)
        except Exception:
            self.stats["rl_invalid"] += 1
            self.stats["rl_time_raw"] += time.perf_counter() - start
            return None

        self.stats["rl_time_raw"] += time.perf_counter() - start
        var = action // 2 + 1
        value = 1 if action % 2 == 1 else -1
        if 1 <= var <= min(solver.inst.n_vars, N_VARS) and solver.assignment[var] == 0:
            return var, value

        self.stats["rl_invalid"] += 1
        return None

    def _static_decision(self, solver: CDCLSolver) -> Optional[tuple[int, int]]:
        for var, value in self.static_order:
            if 1 <= var <= solver.inst.n_vars and solver.assignment[var] == 0:
                return var, value
        return None


def _merge_orders(*orders_and_n_vars) -> LiteralOrder:
    *orders, n_vars = orders_and_n_vars
    merged: LiteralOrder = []
    seen: set[int] = set()
    for order in orders:
        for var, value in order:
            if var in seen:
                continue
            seen.add(var)
            merged.append((var, value))
    for var in range(1, n_vars + 1):
        if var not in seen:
            merged.append((var, 1))
    return merged


def _global_feature_array(global_features: Optional[Iterable[float]]) -> np.ndarray:
    if global_features is None:
        return np.zeros(N_GLOBAL, dtype=np.float32)
    arr = np.array(list(global_features), dtype=np.float32)
    if len(arr) >= N_GLOBAL:
        return arr[:N_GLOBAL]
    return np.pad(arr, (0, N_GLOBAL - len(arr))).astype(np.float32)
