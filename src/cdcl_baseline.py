import time
from typing import Optional, Tuple, List

class SolverStats:
    """Theo dõi hiệu suất thuật toán để vẽ biểu đồ và đánh giá Reward cho AI"""
    def __init__(self):
        self.decisions = 0
        self.conflicts = 0
        self.restarts = 0

class SATInstance:
    """Trình phân tích cú pháp chuẩn DIMACS"""
    def __init__(self, n_vars: int, clauses: List[List[int]]):
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
                if not line or line.startswith("c") or line.startswith("%") or line == "0":
                    continue
                if line.startswith("p"):
                    n_vars = int(line.split()[2])
                    continue
                lits = list(map(int, line.split()))
                if lits and lits[-1] == 0:
                    lits = lits[:-1]
                if lits:
                    clauses.append(lits)
        return cls(n_vars, clauses)

class VSIDS:
    """Variable State Independent Decaying Sum Heuristic"""
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
        self.activity = [a * self.decay for a in self.activity]

    def pick(self, unassigned: List[int]) -> int:
        # Chọn biến có activity cao nhất làm mốc rẽ nhánh
        return max(unassigned, key=lambda v: self.activity[v])

class CDCLSolver:
    """Bộ giải Conflict-Driven Clause Learning chuẩn mực"""
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
        self.stats = SolverStats()  # <- Cực kỳ quan trọng cho môi trường Gym

    def _lit_value(self, lit: int) -> int:
        val = self.assignment[abs(lit)]
        if val == 0: return 0
        return val if lit > 0 else -val

    def _enqueue(self, var: int, value: int, level: int, reason: Optional[int]):
        """Gán biến và lưu lịch sử (API bắt buộc cho Gym Env)"""
        self.assignment[var] = value
        self.decision_level[var] = level
        self.antecedent[var] = reason
        self.trail.append((var, value))

    def _find_initial_units(self):
        """Khởi chạy lan truyền ở mức 0 (API bắt buộc cho Gym Env)"""
        self.unit_propagate()

    def pick_branching_variable(self) -> Optional[Tuple[int, int]]:
        """Giao diện chuẩn cho AI Fallback: Trả về Tuple (Biến, Giá trị)"""
        unassigned = [v for v in range(1, self.inst.n_vars + 1) if self.assignment[v] == 0]
        if not unassigned:
            return None
        var = self.vsids.pick(unassigned)
        self.stats.decisions += 1
        return (var, 1)  # Mặc định Phase Saving là True (1)

    def unit_propagate(self) -> Optional[int]:
        """BCP (Boolean Constraint Propagation) - Cốt lõi tốc độ"""
        while True:
            found_unit = False
            for ci, clause in enumerate(self.clauses):
                unassigned_lits = []
                satisfied = False
                for lit in clause:
                    val = self._lit_value(lit)
                    if val == 1:
                        satisfied = True
                        break
                    if val == 0:
                        unassigned_lits.append(lit)
                
                if satisfied: continue
                if len(unassigned_lits) == 0:
                    return ci # Gặp Conflict
                if len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var = abs(lit)
                    val = 1 if lit > 0 else -1
                    self._enqueue(var, val, self.current_level, ci)
                    found_unit = True
            if not found_unit:
                break
        return None

    def analyze_conflict(self, conflict_ci: int) -> Tuple[List[int], int]:
        """Thuật toán First-UIP (Unique Implication Point)"""
        self.stats.conflicts += 1
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
                if trail_pos < 0: break
                var, _ = self.trail[trail_pos]
                trail_pos -= 1
                if var in seen and self.decision_level[var] == self.current_level:
                    break
            
            counter -= 1
            if counter <= 0:
                if trail_pos + 1 < len(self.trail):
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
        """Quay lui phi tuần tự và dọn dẹp Trail Lim"""
        while self.trail and self.decision_level[self.trail[-1][0]] > level:
            var, _ = self.trail.pop()
            self.assignment[var] = 0
            self.decision_level[var] = 0
            self.antecedent[var] = None
        
        # Cập nhật giới hạn rẽ nhánh (Rất quan trọng cho AI tích hợp)
        while self.trail_lim and self.trail_lim[-1] > len(self.trail):
            self.trail_lim.pop()
            
        self.current_level = level

    def solve(self) -> Tuple[bool, float]:
        """Vòng lặp chính của CDCL Baseline"""
        start = time.time()
        if self.unit_propagate() is not None:
            return False, time.time() - start
        
        while True:
            decision = self.pick_branching_variable()
            if decision is None:
                return True, time.time() - start
            
            var, val = decision
            self.current_level += 1
            self.trail_lim.append(len(self.trail))
            self._enqueue(var, val, self.current_level, None)
            
            while True:
                conflict_ci = self.unit_propagate()
                if conflict_ci is None:
                    break
                
                if self.current_level == 0:
                    return False, time.time() - start
                
                learned, btlevel = self.analyze_conflict(conflict_ci)
                self.backtrack(btlevel)
                self.clauses.append(learned)
                
                if len(learned) > 0:
                    lit = learned[0]
                    val = 1 if lit > 0 else -1
                    self._enqueue(abs(lit), val, self.current_level, len(self.clauses)-1)

def solve_file(filepath: str) -> Tuple[bool, float]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    return solver.solve()
