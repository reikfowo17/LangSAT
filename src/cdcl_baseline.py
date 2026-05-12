
import time
from dataclasses import dataclass
from typing import Optional


class SATInstance:
    def __init__(self, n_vars: int, clauses: list[list[int]]):
        self.n_vars = n_vars
        self.clauses = clauses
        self.n_clauses = len(clauses)

    @classmethod
    def from_dimacs(cls, filepath: str) -> "SATInstance":
        clauses = []
        n_vars = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                # ĐÃ SỬA: Bỏ qua ký tự % và 0 ở cuối file SATLIB
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
        return max(unassigned, key=lambda v: self.activity[v])

class CDCLSolver:
    def __init__(self, instance: SATInstance):
        self.inst = instance
        n = instance.n_vars
        self.assignment = [0] * (n + 1)
        self.decision_level = [0] * (n + 1)
        self.antecedent = [None] * (n + 1)
        self.trail = []
        self.trail_lim = []
        self.clauses = [list(c) for c in instance.clauses]
        self.vsids = VSIDS(n)
        self.current_level = 0
        self.stats = SolveStats()

    def _lit_value(self, lit: int) -> int:
        val = self.assignment[abs(lit)]
        if val == 0: return 0
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
        while True:
            found_unit = False
            for ci, clause in enumerate(self.clauses):
                unassigned = []
                satisfied = False
                for lit in clause:
                    val = self._lit_value(lit)
                    if val == 1:
                        satisfied = True
                        break
                    if val == 0:
                        unassigned.append(lit)
                
                if satisfied: continue
                if len(unassigned) == 0: return ci # Conflict
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    self._assign(abs(lit), 1 if lit > 0 else -1, self.current_level, ci)
                    found_unit = True
            if not found_unit: break
        return None

    def analyze_conflict(self, conflict_ci: int):
        clause = self.clauses[conflict_ci]
        seen = set()
        learned = []
        counter = 0
        trail_pos = len(self.trail) - 1
        
        while True:
            for lit in clause:
                var = abs(lit)
                if var not in seen:
                    seen.add(var)
                    self.vsids.bump(var)
                    if self.decision_level[var] == self.current_level:
                        counter += 1
                    elif self.decision_level[var] > 0:
                        learned.append(lit)
            
            while True:
                var, _ = self.trail[trail_pos]
                trail_pos -= 1
                if var in seen and self.decision_level[var] == self.current_level:
                    break
            
            counter -= 1
            if counter <= 0:
                learned.insert(0, -self.trail[trail_pos+1][1] * self.trail[trail_pos+1][0])
                break
            
            ant_idx = self.antecedent[self.trail[trail_pos+1][0]]
            if ant_idx is None: break
            clause = self.clauses[ant_idx]

        btlevel = 0
        if len(learned) > 1:
            btlevel = max([self.decision_level[abs(l)] for l in learned[1:]])
        
        self.vsids.decay_all()
        return learned, btlevel

    def backtrack(self, level: int):
        while self.trail and self.decision_level[self.trail[-1]] > level:
            var = self.trail.pop()
            self.assignment[var] = 0
            self.decision_level[var] = -1
            self.antecedent[var] = None
        self.current_level = level

    def solve(self) -> tuple[bool, float]:
        start = time.time()
        if self.unit_propagate() is not None: return False, time.time() - start
        
        count = 0
        while True:
            count += 1
            # In debug mỗi 50 bước để tránh tràn màn hình nhưng vẫn theo dõi được tiến độ
            if count % 50 == 0:
                print(f"DEBUG: Step {count}, Level {self.current_level}, Clauses {len(self.clauses)}")
            
            unassigned = [v for v in range(1, self.inst.n_vars + 1) if self.assignment[v] == 0]
            if not unassigned: return True, time.time() - start
            
            var = self.vsids.pick(unassigned)
            self.current_level += 1
            self._assign(var, 1, self.current_level)
            
            while True:
                conflict_ci = self.unit_propagate()
                if conflict_ci is None: break
                if self.current_level == 0: return False, time.time() - start
                
                learned, btlevel = self.analyze_conflict(conflict_ci)
                self.backtrack(btlevel)
                self.clauses.append(learned)
                # Force gán literal vừa học
                lit = learned[0]
                self._assign(abs(lit), 1 if lit > 0 else -1, self.current_level, len(self.clauses)-1)

def solve_file(filepath: str, preferred_literals: Optional[list[int]] = None) -> tuple[bool, float]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    return solver.solve(preferred_literals=preferred_literals)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sat, t = solve_file(sys.argv[1])
        print(f"Result: {'SAT' if sat else 'UNSAT'} in {t:.4f}s")
