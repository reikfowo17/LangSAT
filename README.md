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
