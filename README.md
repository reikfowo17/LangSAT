# LangSAT Reproduce
**Reproduce thực nghiệm bài báo:** LangSAT: A Novel Framework Combining NLP and Reinforcement Learning for SAT Solving (arXiv:2512.04374v1)

## Nhóm thực hiện
| MSSV | Họ và tên | Vai trò |
|------|-----------|---------|
| 24520158 | Lê Gia Bảo | SmartSAT RL (`smartsat_env.py`) |
| 24520159 | Lê Hoàng Gia Bảo | CDCL Baseline (`cdcl_baseline.py`) |
| 24520030 | Lê Đoàn Phúc Thanh | Lang2Logic / NLP (`lang2logic.py`) |
| 23520317 | Trần Đăng Đức | Training + Báo cáo (`training_pipeline.py`, `evaluate.py`) |

---

## Cấu trúc thư mục
```
langsat_reproduce/
├── LangSAT_Reproduce.ipynb   ← Notebook chính
├── requirements.txt
├── README.md
├── src/
│   ├── cdcl_baseline.py       ← CDCL solver + VSIDS heuristic
│   ├── smartsat_env.py        ← Custom Gym environment cho RL agent
│   ├── training_pipeline.py   ← PPO training loop
│   ├── evaluate.py            ← Win rate + solving time evaluation
│   └── lang2logic.py          ← NL → CNF pipeline
├── data/                      ← Dataset uf20-91
└── results/                   ← Output: model, plots, CSV
```

## Kết quả mục tiêu (theo bài báo gốc)
| Chỉ số | Bài báo gốc | Ngưỡng chấp nhận |
|--------|-------------|-----------------|
| Win rate SmartSAT vs Baseline | ~53% | 50% – 56% |
| Median solving time SmartSAT | ~1.02s | 0.9 – 1.15s |
| Median solving time Baseline | ~1.02s | 0.9 – 1.15s |
| SAT success rate | 100% | ≥ 98% |

---

## Hyperparameters (theo bài báo gốc)
- Learning rate: `0.0002`
- Total training steps: `100,000` (1 epoch)
- Train/Test split: `800/200` (80/20)
- Algorithm: `PPO` (Stable-Baselines3)
- Dataset: `uf20-91` (20 variables, 91 clauses, 1000 instances)

---

## Lang2Logic → DIMACS → SAT

`lang2logic.py` chuyển English/propositional text thành CNF. Với biểu thức logic đã ở format bài báo, có thể xuất DIMACS và giải bằng solver trong repo:

```python
from lang2logic import Lang2Logic
from cdcl_baseline import solve_file

pipeline = Lang2Logic()
expr = pipeline.parse_expression("And(Or(A, B), Not(A))")
pipeline.save_dimacs(expr, "results/example.cnf")
sat, seconds = solve_file("results/example.cnf")
```

Chạy từ English text cần `OPENAI_API_KEY`, vì bước NL → logic dùng OpenAI API:

```python
from lang2logic import Lang2Logic

pipeline = Lang2Logic()
result = pipeline.convert("If A then B. A.")
with open("results/from_text.cnf", "w", encoding="utf-8") as f:
    f.write(result["dimacs"]["dimacs"])
```

Phần benchmark SmartSAT trên `uf20-91` vẫn là thực nghiệm riêng để so sánh heuristic với baseline.
