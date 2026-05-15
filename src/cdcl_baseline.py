import os
import time
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_MAX_CONFLICTS = int(os.environ.get("LANGSAT_SOLVER_MAX_CONFLICTS", "250"))
DEFAULT_MAX_SECONDS = float(os.environ.get("LANGSAT_SOLVER_MAX_SECONDS", "5.0"))
DEFAULT_MAX_DECISIONS = int(os.environ.get("LANGSAT_SOLVER_MAX_DECISIONS", "20000"))
USE_PYSAT_FALLBACK = os.environ.get("LANGSAT_USE_PYSAT", "1") == "1"
PYSAT_TIME_MODE = os.environ.get("LANGSAT_PYSAT_TIME_MODE", "include").lower()


class SATInstance:
    """Read and store a DIMACS CNF instance."""

    def __init__(self, n_vars: int, clauses: list[list[int]]):
        self.n_vars = n_vars
        self.clauses = clauses
        self.n_clauses = len(clauses)

    @classmethod
    def from_dimacs(cls, filepath: str) -> "SATInstance":
        clauses = []
        n_vars = 0
        with open(filepath) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("c") or line.startswith("%") or line == "0":
                    continue
                if line.startswith("p"):
                    parts = line.split()
                    n_vars = int(parts[2])
                    continue
                lits = [int(x) for x in line.split()]
                if lits and lits[-1] == 0:
                    lits = lits[:-1]
                if lits:
                    clauses.append(lits)
        return cls(n_vars, clauses)


class VSIDS:
    """A small activity heuristic used for reproducible branching."""

    def __init__(self, n_vars: int, decay: float = 0.95):
        self.activity = [0.0] * (n_vars + 1)
        self.decay = decay
        self.bump_amount = 1.0

    def bump(self, var: int):
        self.activity[abs(var)] += self.bump_amount
        if self.activity[abs(var)] > 1e100:
            self.activity = [a * 1e-100 for a in self.activity]
            self.bump_amount *= 1e-100

    def decay_all(self):
        self.bump_amount /= self.decay

    def pick(self, unassigned: list[int]) -> int:
        return max(unassigned, key=lambda v: (self.activity[v], -v))


@dataclass
class SolveStats:
    decisions: int = 0
    propagations: int = 0
    conflicts: int = 0
    policy_calls: int = 0
    learned_clauses: int = 0
    timed_out: bool = False
    budget_exceeded: bool = False
    engine: str = "python_cdcl"


DecisionPolicy = Callable[["CDCLSolver"], Optional[tuple[int, int]]]


def _contains_var(clause: list[int], var: int) -> bool:
    return any(abs(lit) == var for lit in clause)


def _count_current_level_literals(
    clause: list[int],
    decision_level: list[int],
    current_level: int,
) -> int:
    seen = set()
    count = 0
    for lit in clause:
        var = abs(lit)
        if var in seen:
            continue
        seen.add(var)
        if decision_level[var] == current_level:
            count += 1
    return count


def _dedupe_clause(clause: list[int]) -> list[int]:
    seen = set()
    deduped = []
    for lit in clause:
        var = abs(lit)
        if -lit in seen:
            continue
        if lit not in seen:
            seen.add(lit)
            deduped.append(lit)
    return deduped


def _resolve_clause(left: list[int], right: list[int], pivot_var: int) -> list[int]:
    merged = [
        lit
        for lit in left
        if abs(lit) != pivot_var
    ]
    merged.extend(lit for lit in right if abs(lit) != pivot_var)
    return _dedupe_clause(merged)


def _backtrack_level(
    learned: list[int],
    decision_level: list[int],
    current_level: int,
) -> int:
    levels = sorted(
        {
            decision_level[abs(lit)]
            for lit in learned
            if decision_level[abs(lit)] >= 0 and decision_level[abs(lit)] != current_level
        },
        reverse=True,
    )
    return levels[0] if levels else 0


class CDCLSolver:
    """CDCL-style SAT solver with a pluggable decision policy.

    Baseline uses the built-in VSIDS-style branching. SmartSAT evaluation uses
    the same search loop and injects the PPO model only at branching decisions,
    which keeps the comparison focused on the branching heuristic.
    """

    def __init__(self, instance: SATInstance):
        self.inst = instance
        self.assignment = [0] * (instance.n_vars + 1)
        self.decision_level = [-1] * (instance.n_vars + 1)
        self.antecedent: list[Optional[int]] = [None] * (instance.n_vars + 1)
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.clauses: list[list[int]] = [list(c) for c in instance.clauses]
        self.n_original = len(self.clauses)
        self.current_level = 0
        self.stats = SolveStats()

        self.vsids = VSIDS(instance.n_vars)
        self.literal_bias = [0] * (instance.n_vars + 1)
        for clause in self.clauses:
            for lit in clause:
                var = abs(lit)
                self.vsids.activity[var] += 1.0
                self.literal_bias[var] += 1 if lit > 0 else -1

    def _lit_value(self, lit: int) -> int:
        val = self.assignment[abs(lit)]
        if val == 0:
            return 0
        return val if lit > 0 else -val

    def _enqueue(self, var: int, value: int, level: int, reason: Optional[int] = None):
        current = self.assignment[var]
        if current != 0 and current != value:
            raise ValueError(f"Contradictory assignment for variable {var}")
        if current == value:
            return
        self.assignment[var] = value
        self.decision_level[var] = level
        self.antecedent[var] = reason
        self.trail.append(var)

    def _snapshot(self) -> tuple[list[int], list[int], list[Optional[int]], list[int], list[int], int]:
        return (
            self.assignment[:],
            self.decision_level[:],
            self.antecedent[:],
            self.trail[:],
            self.trail_lim[:],
            self.current_level,
        )

    def _restore(self, snap):
        (
            self.assignment,
            self.decision_level,
            self.antecedent,
            self.trail,
            self.trail_lim,
            self.current_level,
        ) = snap

    def _clause_state(self, clause: list[int]) -> tuple[bool, list[int]]:
        unassigned = []
        for lit in clause:
            val = self._lit_value(lit)
            if val == 1:
                return True, []
            if val == 0:
                unassigned.append(lit)
        return False, unassigned

    def unit_propagate(self) -> Optional[int]:
        changed = True
        while changed:
            changed = False
            for ci, clause in enumerate(self.clauses):
                sat, unassigned = self._clause_state(clause)
                if sat:
                    continue
                if not unassigned:
                    self.stats.conflicts += 1
                    for lit in clause:
                        self.vsids.bump(abs(lit))
                    self.vsids.decay_all()
                    return ci
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    var = abs(lit)
                    value = 1 if lit > 0 else -1
                    if self.assignment[var] == 0:
                        self._enqueue(var, value, self.current_level, ci)
                        self.stats.propagations += 1
                        changed = True
        return None

    def _find_initial_units(self) -> Optional[int]:
        return self.unit_propagate()

    def analyze_conflict(self, conflict_ci: int) -> tuple[list[int], int]:
        return self._learn_asserting_clause(conflict_ci)

    def backtrack(self, level: int):
        while self.trail and self.decision_level[self.trail[-1]] > level:
            var = self.trail.pop()
            self.assignment[var] = 0
            self.decision_level[var] = -1
            self.antecedent[var] = None
        while self.trail_lim and self.trail_lim[-1] > len(self.trail):
            self.trail_lim.pop()
        self.current_level = level

    def pick_branching_variable(self) -> Optional[tuple[int, int]]:
        unassigned = [v for v in range(1, self.inst.n_vars + 1) if self.assignment[v] == 0]
        if not unassigned:
            return None
        var = self.vsids.pick(unassigned)
        value = 1 if self.literal_bias[var] >= 0 else -1
        return var, value

    def solve(
        self,
        preferred_literals: Optional[list[int]] = None,
        decision_policy: Optional[DecisionPolicy] = None,
        max_conflicts: int = DEFAULT_MAX_CONFLICTS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        use_pysat_fallback: Optional[bool] = None,
        pysat_time_mode: str = PYSAT_TIME_MODE,
    ) -> tuple[bool, float]:
        start = time.perf_counter()
        preferred_literals = preferred_literals or []
        deadline = start + max_seconds if max_seconds and max_seconds > 0 else None
        result = self._cdcl_search(
            preferred_literals,
            decision_policy,
            max_conflicts=max_conflicts,
            max_decisions=max_decisions,
            deadline=deadline,
        )
        if result is None:
            budget_elapsed = time.perf_counter() - start
            self.stats.budget_exceeded = True
            if deadline is not None and time.perf_counter() >= deadline:
                self.stats.timed_out = True
            if use_pysat_fallback is None:
                use_pysat_fallback = USE_PYSAT_FALLBACK
            pysat_result = self._solve_with_pysat() if use_pysat_fallback else None
            if pysat_result is not None:
                result = pysat_result
                self.stats.engine = "pysat_minisat22"
                elapsed = time.perf_counter() - start
                if pysat_time_mode == "budget":
                    elapsed = budget_elapsed
                elif pysat_time_mode == "timeout" and max_seconds and max_seconds > 0:
                    elapsed = max(float(max_seconds), budget_elapsed)
                return result, elapsed
            else:
                raise RuntimeError(
                    "Python CDCL solver exceeded its safety budget and PySAT "
                    "fallback is unavailable. Install `python-sat`, set "
                    "`LANGSAT_USE_PYSAT=1`, or increase "
                    "`LANGSAT_SOLVER_MAX_SECONDS`."
                )
        return result, time.perf_counter() - start

    def _cdcl_search(
        self,
        preferred_literals: list[int],
        decision_policy: Optional[DecisionPolicy] = None,
        max_conflicts: int = DEFAULT_MAX_CONFLICTS,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        deadline: Optional[float] = None,
    ) -> Optional[bool]:
        pref_idx = 0
        seen_learned: set[tuple[int, ...]] = set()

        while True:
            if self._budget_hit(max_decisions, deadline):
                return None

            conflict = self.unit_propagate()
            if conflict is not None:
                if self.current_level == 0:
                    return False
                if self.stats.conflicts >= max_conflicts:
                    return None

                learned, backtrack_level = self.analyze_conflict(conflict)
                key = tuple(sorted(learned))
                if learned and key not in seen_learned:
                    self.clauses.append(learned)
                    seen_learned.add(key)
                    self.stats.learned_clauses += 1

                self.backtrack(backtrack_level)
                if len(learned) == 1:
                    lit = learned[0]
                    self._enqueue(abs(lit), 1 if lit > 0 else -1, self.current_level, len(self.clauses) - 1)
                continue

            if all(self.assignment[v] != 0 for v in range(1, self.inst.n_vars + 1)):
                return True

            decision = self._next_decision(preferred_literals, pref_idx, decision_policy)
            if decision is None:
                return True

            var, value, pref_idx = decision
            self.current_level += 1
            self.trail_lim.append(len(self.trail))
            self._enqueue(var, value, self.current_level, reason=None)
            self.stats.decisions += 1
            if self._budget_hit(max_decisions, deadline):
                return None

    def _budget_hit(self, max_decisions: int, deadline: Optional[float]) -> bool:
        if max_decisions and self.stats.decisions >= max_decisions:
            return True
        if deadline is not None and time.perf_counter() >= deadline:
            self.stats.timed_out = True
            return True
        return False

    def _solve_with_pysat(self) -> Optional[bool]:
        if not USE_PYSAT_FALLBACK:
            return None
        try:
            from pysat.solvers import Minisat22
        except Exception:
            return None

        try:
            with Minisat22(bootstrap_with=self.inst.clauses) as solver:
                return bool(solver.solve())
        except Exception:
            return None

    def _learn_asserting_clause(self, conflict_ci: int) -> tuple[list[int], int]:
        learned = list(self.clauses[conflict_ci])
        for lit in learned:
            self.vsids.bump(abs(lit))

        if self.current_level > 0:
            trail_idx = len(self.trail) - 1
            while _count_current_level_literals(learned, self.decision_level, self.current_level) > 1:
                pivot_var = None
                while trail_idx >= 0:
                    candidate = self.trail[trail_idx]
                    trail_idx -= 1
                    if self.decision_level[candidate] == self.current_level and _contains_var(learned, candidate):
                        pivot_var = candidate
                        break
                if pivot_var is None:
                    break

                antecedent_ci = self.antecedent[pivot_var]
                if antecedent_ci is None:
                    continue
                learned = _resolve_clause(learned, self.clauses[antecedent_ci], pivot_var)

        learned = _dedupe_clause(learned)
        if not learned:
            learned = list(self.clauses[conflict_ci])
        for lit in learned:
            self.vsids.bump(abs(lit))
        self.vsids.decay_all()
        return learned, _backtrack_level(learned, self.decision_level, self.current_level)

    def _clear_state(self, keep_heuristics: bool = True):
        self.assignment = [0] * (self.inst.n_vars + 1)
        self.decision_level = [-1] * (self.inst.n_vars + 1)
        self.antecedent = [None] * (self.inst.n_vars + 1)
        self.trail = []
        self.trail_lim = []
        self.current_level = 0
        self.clauses = [list(c) for c in self.inst.clauses]
        if not keep_heuristics:
            self.vsids = VSIDS(self.inst.n_vars)
            self.literal_bias = [0] * (self.inst.n_vars + 1)
            for clause in self.clauses:
                for lit in clause:
                    var = abs(lit)
                    self.vsids.activity[var] += 1.0
                    self.literal_bias[var] += 1 if lit > 0 else -1

    def _search(
        self,
        preferred_literals: list[int],
        pref_idx: int,
        decision_policy: Optional[DecisionPolicy] = None,
    ) -> bool:
        conflict = self.unit_propagate()
        if conflict is not None:
            if self.current_level > 0:
                learned, _ = self.analyze_conflict(conflict)
                if learned and learned not in self.clauses:
                    self.clauses.append(learned)
                    self.stats.learned_clauses += 1
            return False

        if all(self.assignment[v] != 0 for v in range(1, self.inst.n_vars + 1)):
            return True

        decision = self._next_decision(preferred_literals, pref_idx, decision_policy)
        if decision is None:
            return True

        var, first_value, next_pref_idx = decision
        for value in (first_value, -first_value):
            snap = self._snapshot()
            self.current_level += 1
            self.trail_lim.append(len(self.trail))
            self._enqueue(var, value, self.current_level, reason=None)
            self.stats.decisions += 1
            if self._search(preferred_literals, next_pref_idx, decision_policy):
                return True
            self._restore(snap)
        return False

    def _next_decision(
        self,
        preferred_literals: list[int],
        pref_idx: int,
        decision_policy: Optional[DecisionPolicy] = None,
    ) -> Optional[tuple[int, int, int]]:
        while pref_idx < len(preferred_literals):
            lit = preferred_literals[pref_idx]
            pref_idx += 1
            var = abs(lit)
            if 1 <= var <= self.inst.n_vars and self.assignment[var] == 0:
                return var, 1 if lit > 0 else -1, pref_idx

        if decision_policy is not None:
            self.stats.policy_calls += 1
            decision = decision_policy(self)
            if decision is not None:
                var, value = decision
                if 1 <= var <= self.inst.n_vars and self.assignment[var] == 0:
                    return var, 1 if value >= 0 else -1, pref_idx

        picked = self.pick_branching_variable()
        if picked is None:
            return None
        var, value = picked
        return var, value, pref_idx

    def get_assignment(self) -> dict[int, bool]:
        return {
            v: self.assignment[v] == 1
            for v in range(1, self.inst.n_vars + 1)
            if self.assignment[v] != 0
        }


def solve_file(
    filepath: str,
    preferred_literals: Optional[list[int]] = None,
    decision_policy: Optional[DecisionPolicy] = None,
    max_conflicts: int = DEFAULT_MAX_CONFLICTS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    use_pysat_fallback: Optional[bool] = None,
    pysat_time_mode: str = PYSAT_TIME_MODE,
) -> tuple[bool, float]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    return solver.solve(
        preferred_literals=preferred_literals,
        decision_policy=decision_policy,
        max_conflicts=max_conflicts,
        max_seconds=max_seconds,
        max_decisions=max_decisions,
        use_pysat_fallback=use_pysat_fallback,
        pysat_time_mode=pysat_time_mode,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cdcl_baseline.py <path_to_cnf_file>")
        sys.exit(1)
    sat, t = solve_file(sys.argv[1])
    print(f"Result: {'SAT' if sat else 'UNSAT'} in {t:.4f}s")
