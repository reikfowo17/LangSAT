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
│   ├── policy.py              ← Graph message-passing PPO feature extractor
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

`lang2logic.py` chuyển English/propositional text thành CNF. Bước English → logic dùng OpenAI API. Nếu input đã ở format logic của bài báo (`And(...)`, `Or(...)`, `Not(...)`, `Implies(...)`, `Equivalent(...)`) thì không cần API:

```python
from lang2logic import Lang2Logic
from cdcl_baseline import solve_file

pipeline = Lang2Logic()
expr = pipeline.parse_expression("And(Or(A, B), Not(A))")
pipeline.save_dimacs(expr, "results/example.cnf")
sat, seconds = solve_file("results/example.cnf")
```

Chạy từ English text cần `OPENAI_API_KEY`:

```python
from lang2logic import Lang2Logic

pipeline = Lang2Logic()
result = pipeline.convert("If A then B. A.")
with open("results/from_text.cnf", "w", encoding="utf-8") as f:
    f.write(result["dimacs"]["dimacs"])
```

Phần benchmark SmartSAT trên `uf20-91` vẫn là thực nghiệm riêng để so sánh heuristic với baseline.

## Ghi chú reproduce

- Đường chạy mặc định là paper-like và nghiêm ngặt: reward paper, total solving time, bắt buộc SATfeatPy, không có feature tự tính thay thế.
- Split dataset mặc định là `sorted` để giữ tương thích notebook cũ. Có thể đặt `LANGSAT_SPLIT_STRATEGY=shuffled` và `LANGSAT_SPLIT_SEED=42`; metadata được lưu vào `data_split.json`.
- SmartSAT dùng 48 global SAT features từ SATfeatPy/SATzilla. Cần clone SATfeatPy và set `LANGSAT_SATFEATPY_DIR`; nếu thiếu hoặc lỗi, training/evaluation sẽ dừng.
- SmartSAT policy mặc định dùng `SmartSATGraphExtractor` trong `src/policy.py`: PyTorch bipartite message passing trên signed clause-variable graph, fuse với assignment state, clause state và 48 SATfeatPy global features. Không cần `torch-geometric`.
- `satfeat_adapter.py` normalize DIMACS trước khi gọi SATfeatPy để tránh lỗi parser với header SATLIB có nhiều khoảng trắng và cache riêng cho SATfeatPy. Strict reproduction mặc định bật local-search probing (`LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH=1`) để lấy đủ 48 SATzilla features; nếu Kaggle thiếu `ubcsat`, đặt biến này thành `0` chỉ cho diagnostic partial-feature run.
- Mặc định SmartSAT dùng `LANGSAT_POLICY_MODE=rl`: PPO chọn trực tiếp branching action, reward paper là delta của số clause satisfied trừ unsatisfied nên cumulative episode reward bị chặn bởi final clause score, tối đa 91 trên `uf20-91`.
- SmartSAT benchmark hiện cố định profile `uf20-91` (20 variables, 91 clauses). CNF khác profile sẽ bị từ chối rõ ràng thay vì âm thầm truncate observation/action space.
- Baseline dùng CDCL-style search với VSIDS, conflict learning và backtracking. Pure reproduce chỉ dùng Python CDCL; nếu solver vượt budget, hãy tăng `LANGSAT_SOLVER_MAX_SECONDS`, `LANGSAT_SOLVER_MAX_CONFLICTS`, hoặc `LANGSAT_SOLVER_MAX_DECISIONS` thay vì dùng engine khác cho số liệu báo cáo.
- `evaluate.py` lưu thêm diagnostic metrics như invalid-action rate, policy time per call, decision/conflict ratio SmartSAT-vs-baseline và budget exit rate để giải thích sai lệch reproduction thay vì chỉ nhìn win rate.
- Kết quả thời gian luôn là raw runtime. Repo không còn scale thời gian để khớp mốc paper.
- `src/end_to_end.py` không còn là entrypoint; end-to-end được tích hợp vào `lang2logic.py` để bám cấu trúc module của repo.

## Chạy reproduction

```powershell
git clone https://github.com/bprovanbessell/SATfeatPy.git D:\tools\SATfeatPy
$env:LANGSAT_SATFEATPY_DIR="D:\tools\SATfeatPy"
python src\training_pipeline.py
python src\evaluate.py
```

Trên Kaggle, dùng notebook `notebooks/LangSAT_Kaggle_Reproduce.ipynb`; cell đầu đã clone repo, clone SATfeatPy và set các biến môi trường cần thiết.
