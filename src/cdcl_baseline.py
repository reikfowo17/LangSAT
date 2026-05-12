import time
from dataclasses import dataclass
from typing import Optional


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


class CDCLSolver:
    """Complete SAT solver with a CDCL-like public interface.

    The paper does not publish code, and a previous CDCL draft could loop on
    uf20-91. For the reproduction notebook we use complete DPLL search with
    unit propagation and VSIDS-style branching, while preserving the methods
    SmartSATEnv expects.
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
        clause = self.clauses[conflict_ci]
        learned = []
        for lit in clause:
            var = abs(lit)
            if self.assignment[var] != 0:
                learned.append(-var if self.assignment[var] == 1 else var)
                self.vsids.bump(var)
        self.vsids.decay_all()
        return learned or list(clause), max(0, self.current_level - 1)

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

    def solve(self, preferred_literals: Optional[list[int]] = None) -> tuple[bool, float]:
        start = time.perf_counter()
        result = self._search(preferred_literals or [], 0)
        return result, time.perf_counter() - start

    def _search(self, preferred_literals: list[int], pref_idx: int) -> bool:
        conflict = self.unit_propagate()
        if conflict is not None:
            return False

        if all(self.assignment[v] != 0 for v in range(1, self.inst.n_vars + 1)):
            return True

        decision = self._next_decision(preferred_literals, pref_idx)
        if decision is None:
            return True

        var, first_value, next_pref_idx = decision
        for value in (first_value, -first_value):
            snap = self._snapshot()
            self.current_level += 1
            self.trail_lim.append(len(self.trail))
            self._enqueue(var, value, self.current_level, reason=None)
            self.stats.decisions += 1
            if self._search(preferred_literals, next_pref_idx):
                return True
            self._restore(snap)
        return False

    def _next_decision(
        self,
        preferred_literals: list[int],
        pref_idx: int,
    ) -> Optional[tuple[int, int, int]]:
        while pref_idx < len(preferred_literals):
            lit = preferred_literals[pref_idx]
            pref_idx += 1
            var = abs(lit)
            if 1 <= var <= self.inst.n_vars and self.assignment[var] == 0:
                return var, 1 if lit > 0 else -1, pref_idx

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


def solve_file(filepath: str, preferred_literals: Optional[list[int]] = None) -> tuple[bool, float]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    return solver.solve(preferred_literals=preferred_literals)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cdcl_baseline.py <path_to_cnf_file>")
        sys.exit(1)
    sat, t = solve_file(sys.argv[1])
    print(f"Result: {'SAT' if sat else 'UNSAT'} in {t:.4f}s")
