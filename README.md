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
│   ├── competitive_advisor.py ← Paper-inspired one-shot branching advisor
│   ├── evaluate.py            ← Win rate + solving time evaluation
│   └── lang2logic.py          ← NL → CNF pipeline
├── data/                      ← Dataset uf20-91
└── results/                   ← Output: model, plots, CSV
```

## Kết quả mục tiêu
| Profile | Mục tiêu | Cách đo |
|---------|----------|---------|
| `competitive` | Tối đa win rate SmartSAT vs baseline hiện tại của repo | `uf20-91`, raw total time |
| `paper_like` | Đối chiếu bài báo gốc: ~53% win rate, ~1.02s median | `uf20-91`, raw total time |
| Cả hai | SAT success rate cao và không scale thời gian | `eval_results.csv`, `metrics.json` |

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

## Ghi chú SmartSAT

- Profile mặc định của evaluation là `competitive`: SmartSAT dùng one-shot occurrence/Jeroslow-Wang/MOMS branching advisor trong `competitive_advisor.py`. Hướng này lấy cảm hứng từ các nghiên cứu SAT guidance hiện đại: giảm inference trong hot CDCL loop để thắng trên raw total time.
- `competitive` vẫn giữ baseline repo nguyên trạng để so winrate công bằng: baseline là CDCL + VSIDS trong `cdcl_baseline.py`; SmartSAT là CDCL cùng lõi solver nhưng heuristic branch khác.
- `competitive` mặc định không cần PPO model và không cần SATfeatPy (`feature_backend=none_static_advisor`). Nếu muốn chạy lại PPO online kiểu paper, đặt `LANGSAT_POLICY_MODE=rl`. Nếu muốn competitive thử RL trước static advisor, đặt thêm `LANGSAT_COMPETITIVE_ENABLE_RL=1`; mặc định là `0` để tránh overhead và tăng raw-time winrate.
- `evaluate.py` lưu thêm advisor metrics: static route, baseline fallback route, RL route, invalid action rate, policy overhead, decision/conflict ratio và budget exit rate.
- Kết quả thời gian luôn là raw runtime. Repo không còn scale thời gian để khớp mốc paper.

## Ghi chú reproduce paper-like

- Đường chạy `paper_like` vẫn giữ reward paper, total solving time, bắt buộc SATfeatPy, không có feature tự tính thay thế.
- Split dataset mặc định là `sorted` để giữ tương thích notebook cũ. Có thể đặt `LANGSAT_SPLIT_STRATEGY=shuffled` và `LANGSAT_SPLIT_SEED=42`; metadata được lưu vào `data_split.json`.
- SmartSAT dùng 48 global SAT features từ SATfeatPy/SATzilla. Cần clone SATfeatPy và set `LANGSAT_SATFEATPY_DIR`; nếu thiếu hoặc lỗi, training/evaluation sẽ dừng.
- SmartSAT policy mặc định dùng `SmartSATGraphExtractor` trong `src/policy.py`: PyTorch bipartite message passing trên signed clause-variable graph, fuse với assignment state, clause state và 48 SATfeatPy global features. Không cần `torch-geometric`.
- `satfeat_adapter.py` normalize DIMACS trước khi gọi SATfeatPy để tránh lỗi parser với header SATLIB có nhiều khoảng trắng và cache riêng cho SATfeatPy. Strict reproduction mặc định bật local-search probing (`LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH=1`) để lấy đủ 48 SATzilla features; nếu Kaggle thiếu `ubcsat`, đặt biến này thành `0` chỉ cho diagnostic partial-feature run.
- PPO training dùng reward paper là delta của số clause satisfied trừ unsatisfied nên cumulative episode reward bị chặn bởi final clause score, tối đa 91 trên `uf20-91`.
- SmartSAT benchmark hiện cố định profile `uf20-91` (20 variables, 91 clauses). CNF khác profile sẽ bị từ chối rõ ràng thay vì âm thầm truncate observation/action space.
- Baseline dùng CDCL-style search với VSIDS, conflict learning và backtracking. Pure reproduce chỉ dùng Python CDCL; nếu solver vượt budget, hãy tăng `LANGSAT_SOLVER_MAX_SECONDS`, `LANGSAT_SOLVER_MAX_CONFLICTS`, hoặc `LANGSAT_SOLVER_MAX_DECISIONS` thay vì dùng engine khác cho số liệu báo cáo.
- `src/end_to_end.py` không còn là entrypoint; end-to-end được tích hợp vào `lang2logic.py` để bám cấu trúc module của repo.

## Chạy reproduction

```powershell
git clone https://github.com/bprovanbessell/SATfeatPy.git D:\tools\SATfeatPy
$env:LANGSAT_SATFEATPY_DIR="D:\tools\SATfeatPy"
# Competitive advisor không cần model PPO nếu LANGSAT_COMPETITIVE_ENABLE_RL=0.
python src\evaluate.py

# Paper-like PPO run:
$env:LANGSAT_POLICY_MODE="rl"
python src\training_pipeline.py
python src\evaluate.py
```

Trên Kaggle, dùng notebook `notebooks/LangSAT_Kaggle_Reproduce.ipynb`; cell đầu đã clone repo, clone SATfeatPy và set các biến môi trường cần thiết.
