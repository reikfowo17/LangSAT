import time
from collections import defaultdict
from typing import Optional

class SATInstance:
    """Đọc và lưu trữ một instance CNF theo định dạng DIMACS."""

    def __init__(self, n_vars: int, clauses: list[list[int]]):
        self.n_vars = n_vars
        self.clauses = clauses          # list of list[int], âm = negation
        self.n_clauses = len(clauses)

    @classmethod
    def from_dimacs(cls, filepath: str) -> "SATInstance":
        clauses = []
        n_vars = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                if line.startswith("p"):
                    parts = line.split()
                    n_vars = int(parts[2])
                    continue
                lits = list(map(int, line.split()))
                if lits and lits[-1] == 0:
                    lits = lits[:-1]
                if lits:
                    clauses.append(lits)
        return cls(n_vars, clauses)


class VSIDS:

    def __init__(self, n_vars: int, decay: float = 0.95):
        self.activity = [0.0] * (n_vars + 1)   # 1-indexed
        self.decay = decay
        self.bump_amount = 1.0
        self._conflicts_since_decay = 0
        self._decay_interval = 1               # decay sau mỗi conflict

    def bump(self, var: int):
        self.activity[abs(var)] += self.bump_amount
        # Rescale nếu quá lớn
        if self.activity[abs(var)] > 1e100:
            self.activity = [a * 1e-100 for a in self.activity]
            self.bump_amount *= 1e-100

    def decay_all(self):
        self.activity = [a * self.decay for a in self.activity]

    def pick(self, unassigned: list[int]) -> int:
        """Chọn biến chưa gán có activity cao nhất."""
        return max(unassigned, key=lambda v: self.activity[v])

class CDCLSolver:
    def __init__(self, instance: SATInstance):
        self.inst = instance
        n = instance.n_vars

        # Assignment: 0 = unassigned, 1 = True, -1 = False
        self.assignment = [0] * (n + 1)
        self.decision_level = [0] * (n + 1)
        self.antecedent = [None] * (n + 1)   # clause index gây ra implication

        self.trail = []           # list[(var, value)] theo thứ tự gán
        self.trail_lim = []       # trail index tại mỗi decision level

        # Learned clauses
        self.clauses = [list(c) for c in instance.clauses]
        self.n_original = len(self.clauses)

        # Watch literals (2-watched scheme đơn giản)
        self.watches: dict[int, list[int]] = defaultdict(list)
        self._init_watches()

        self.vsids = VSIDS(n)
        self.current_level = 0


    def _init_watches(self):
        """Mỗi clause được watch bởi 2 literal đầu tiên."""
        self.watches.clear()
        for ci, clause in enumerate(self.clauses):
            if len(clause) >= 1:
                self.watches[clause[0]].append(ci)
            if len(clause) >= 2:
                self.watches[clause[1]].append(ci)

    # ---- Unit Propagation ----

    def _lit_value(self, lit: int) -> int:
        """Trả về giá trị hiện tại của literal: 1=True, -1=False, 0=Unassigned."""
        val = self.assignment[abs(lit)]
        if val == 0:
            return 0
        return val if lit > 0 else -val

    def _assign(self, var: int, value: int, level: int, antecedent=None):
        self.assignment[var] = value
        self.decision_level[var] = level
        self.antecedent[var] = antecedent
        self.trail.append((var, value))

    def unit_propagate(self) -> Optional[int]:
        queue = []
        for ci, clause in enumerate(self.clauses):
            unsat, unit_lit = self._check_clause_unit(clause)
            if unsat:
                return ci
            if unit_lit is not None:
                queue.append((unit_lit, ci))

        i = 0
        while i < len(queue):
            lit, antecedent_ci = queue[i]
            i += 1
            var = abs(lit)
            val = 1 if lit > 0 else -1

            if self._lit_value(lit) == 1:
                continue   # đã thỏa
            if self._lit_value(lit) == -1:
                return antecedent_ci   # conflict

            self._assign(var, val, self.current_level, antecedent_ci)

            # Kiểm tra clauses bị ảnh hưởng
            for ci, clause in enumerate(self.clauses):
                if -lit in clause:
                    unsat, unit_lit2 = self._check_clause_unit(clause)
                    if unsat:
                        return ci
                    if unit_lit2 is not None:
                        queue.append((unit_lit2, ci))

        return None

    def _check_clause_unit(self, clause: list[int]):
        unassigned = []
        for lit in clause:
            v = self._lit_value(lit)
            if v == 1:
                return False, None   # clause đã thỏa
            if v == 0:
                unassigned.append(lit)
        if len(unassigned) == 0:
            return True, None    # conflict
        if len(unassigned) == 1:
            return False, unassigned[0]  # unit
        return False, None

    # ---- Conflict Analysis ----

    def analyze_conflict(self, conflict_clause_idx: int):
        clause = list(self.clauses[conflict_clause_idx])
        seen = set()
        learned = []
        counter = 0
        current_level = self.current_level
        trail_pos = len(self.trail) - 1

        while True:
            for lit in clause:
                var = abs(lit)
                if var in seen:
                    continue
                seen.add(var)
                self.vsids.bump(var)
                if self.decision_level[var] == current_level:
                    counter += 1
                elif self.decision_level[var] > 0:
                    learned.append(-lit if self.assignment[var] == 1 else lit)

            # Đi ngược trail để tìm literal ở current level tiếp theo
            while trail_pos >= 0:
                var, _ = self.trail[trail_pos]
                trail_pos -= 1
                if var in seen and self.decision_level[var] == current_level:
                    break

            counter -= 1
            if counter <= 0:
                lit_to_add = -self.trail[trail_pos + 1][0] if self.assignment[self.trail[trail_pos + 1][0]] == 1 \
                    else self.trail[trail_pos + 1][0]
                learned = [lit_to_add] + learned
                break

            # Resolve với antecedent
            var, _ = self.trail[trail_pos + 1]
            ant = self.antecedent[var]
            if ant is not None:
                clause = self.clauses[ant]
            else:
                break

        # Backtrack level = max decision level trong learned (trừ current)
        if len(learned) == 1:
            btlevel = 0
        else:
            levels = [self.decision_level[abs(l)] for l in learned[1:]]
            btlevel = max(levels) if levels else 0

        self.vsids.decay_all()
        return learned, btlevel

    # ---- Backtrack ----

    def backtrack(self, level: int):
        while self.trail and self.decision_level[self.trail[-1][0]] > level:
            var, _ = self.trail.pop()
            self.assignment[var] = 0
            self.decision_level[var] = 0
            self.antecedent[var] = None
        # Cắt trail_lim
        while self.trail_lim and self.trail_lim[-1] > len(self.trail):
            self.trail_lim.pop()
        self.current_level = level

    # ---- Pick Branching Variable ----

    def pick_branching_variable(self) -> Optional[tuple[int, int]]:
        unassigned = [v for v in range(1, self.inst.n_vars + 1)
                      if self.assignment[v] == 0]
        if not unassigned:
            return None
        var = self.vsids.pick(unassigned)
        return var, 1   # mặc định gán True

    # ---- Main Solve Loop ----

    def solve(self) -> tuple[bool, float]:
        start = time.time()

        # Unit propagation ban đầu (level 0)
        conflict_ci = self.unit_propagate()
        if conflict_ci is not None:
            return False, time.time() - start

        while True:
            decision = self.pick_branching_variable()
            if decision is None:
                # Tất cả biến đã gán → SAT
                return True, time.time() - start

            var, val = decision
            self.current_level += 1
            self.trail_lim.append(len(self.trail))
            self._assign(var, val, self.current_level, antecedent=None)

            while True:
                conflict_ci = self.unit_propagate()
                if conflict_ci is None:
                    break   # No conflict → tiếp tục pick

                if self.current_level == 0:
                    return False, time.time() - start

                learned, btlevel = self.analyze_conflict(conflict_ci)
                self.backtrack(btlevel)

                # Thêm learned clause
                self.clauses.append(learned)
                ci_new = len(self.clauses) - 1

                # Force unit propagation của learned clause
                if len(learned) == 1:
                    unit_lit = learned[0]
                    var2 = abs(unit_lit)
                    val2 = 1 if unit_lit > 0 else -1
                    if self.assignment[var2] == 0:
                        self._assign(var2, val2, self.current_level, ci_new)

    def get_assignment(self) -> dict[int, bool]:
        return {v: (self.assignment[v] == 1)
                for v in range(1, self.inst.n_vars + 1)
                if self.assignment[v] != 0}


def solve_file(filepath: str) -> tuple[bool, float]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    return solver.solve()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cdcl_baseline.py <path_to_cnf_file>")
        sys.exit(1)
    sat, t = solve_file(sys.argv[1])
    print(f"Result: {'SAT' if sat else 'UNSAT'} in {t:.4f}s")
