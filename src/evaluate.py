import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from cdcl_baseline import SATInstance, CDCLSolver, solve_file
from smartsat_env import SmartSATEnv, N_VARS, N_CLAUSES


OUTPUT_DIR  = "/kaggle/working/results"
MODEL_PATH  = "/kaggle/working/results/smartsat_model"
SPLIT_PATH  = "/kaggle/working/results/data_split.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def solve_with_smartsat(filepath: str, model: PPO) -> tuple[bool, float]:
    env = SmartSATEnv([filepath])
    obs, _ = env.reset()

    start = time.time()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    elapsed = time.time() - start
    sat = info.get("sat", False)
    return sat, elapsed

def evaluate(test_files: list[str], model: PPO) -> pd.DataFrame:
    results = []
    n = len(test_files)

    print(f" EVALUATION — {n} instances")

    for i, filepath in enumerate(test_files):
        # 1. CDCL Baseline
        baseline_sat, baseline_time = solve_file(filepath)

        # 2. SmartSAT
        smartsat_sat, smartsat_time = solve_with_smartsat(filepath, model)

        results.append({
            "file": os.path.basename(filepath),
            "instance_idx": i,
            "baseline_sat": baseline_sat,
            "baseline_time": baseline_time,
            "smartsat_sat": smartsat_sat,
            "smartsat_time": smartsat_time,
        })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:>3}/{n}] Baseline: {baseline_time:.4f}s | SmartSAT: {smartsat_time:.4f}s")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Eval] Results saved → {csv_path}")
    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)

    smartsat_wins  = (df["smartsat_time"] < df["baseline_time"]).sum()
    baseline_wins  = (df["baseline_time"] < df["smartsat_time"]).sum()
    ties           = (df["smartsat_time"] == df["baseline_time"]).sum()

    win_rate = smartsat_wins / n * 100

    metrics = {
        "n_instances"      : n,
        "smartsat_wins"    : int(smartsat_wins),
        "baseline_wins"    : int(baseline_wins),
        "ties"             : int(ties),
        "win_rate_pct"     : round(win_rate, 2),
        "median_smartsat"  : round(float(df["smartsat_time"].median()), 4),
        "median_baseline"  : round(float(df["baseline_time"].median()), 4),
        "mean_smartsat"    : round(float(df["smartsat_time"].mean()), 4),
        "mean_baseline"    : round(float(df["baseline_time"].mean()), 4),
        "sat_rate_smartsat": round(df["smartsat_sat"].mean() * 100, 2),
        "sat_rate_baseline": round(df["baseline_sat"].mean() * 100, 2),
    }

    # In bảng kết quả
    print("  KẾT QUẢ EVALUATION")
    print(f"  Số instances test    : {metrics['n_instances']}")
    print(f"  SmartSAT thắng       : {metrics['smartsat_wins']} ({metrics['win_rate_pct']}%)")
    print(f"  Baseline thắng       : {metrics['baseline_wins']}")
    print(f"  Hòa                  : {metrics['ties']}")
    print(f"  Median SmartSAT      : {metrics['median_smartsat']}s")
    print(f"  Median Baseline      : {metrics['median_baseline']}s")
    print(f"  [Bài báo gốc]        : ~53% win rate, ~1.02s median")
    print("="*55)

    # So sánh với bài báo
    paper_winrate = 53.0
    paper_median  = 1.02
    print(f"\n  Sai lệch win rate   : {abs(metrics['win_rate_pct'] - paper_winrate):.2f}%")
    print(f"  Sai lệch median ST  : {abs(metrics['median_smartsat'] - paper_median):.4f}s")
    print(f"  Sai lệch median BSL : {abs(metrics['median_baseline'] - paper_median):.4f}s")

    # Lưu metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {OUTPUT_DIR}/metrics.json")

    return metrics


def plot_solving_times(df: pd.DataFrame, metrics: dict):
    fig, ax = plt.subplots(figsize=(14, 5))

    x = df["instance_idx"].values

    # Baseline (vàng/cam)
    ax.plot(x, df["baseline_time"].values,
            color="#F5A623", linewidth=0.8, alpha=0.85,
            label="Baseline Times")

    # SmartSAT (xanh lá)
    ax.plot(x, df["smartsat_time"].values,
            color="#4CAF50", linewidth=0.8, alpha=0.85,
            label="SmartSAT Times")

    # Median lines (dashed)
    ax.axhline(metrics["median_baseline"], color="#F5A623", linestyle="--",
               linewidth=1.2, label=f"Baseline Median — {metrics['median_baseline']}s")
    ax.axhline(metrics["median_smartsat"], color="#4CAF50", linestyle="--",
               linewidth=1.2, label=f"SmartSAT Median — {metrics['median_smartsat']}s")

    ax.set_xlabel("Test Set Problem Number", fontsize=11)
    ax.set_ylabel("Time Taken (seconds)", fontsize=11)
    ax.set_title("Comparison of Baseline Times and SmartSAT Times for uf20-91 Test Set",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-2, len(df) + 2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "solving_time_comparison.png")
    plt.savefig(path, dpi=180)
    plt.show()
    print(f"[Plot] Solving time plot saved → {path}")


def plot_time_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, color, label in [
        (axes[0], "baseline_time", "#F5A623", "Baseline CDCL"),
        (axes[1], "smartsat_time", "#4CAF50", "SmartSAT"),
    ]:
        ax.hist(df[col], bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(df[col].median(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median: {df[col].median():.4f}s")
        ax.set_title(f"{label} — Solving Time Distribution")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "time_distribution.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Plot] Distribution plot saved → {path}")

if __name__ == "__main__":
    # 1. Load test file list
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        test_files = split["test"]
        print(f"[Eval] Loaded {len(test_files)} test files from split.")
    else:
        raise FileNotFoundError(
            "Chưa có data_split.json. Hãy chạy training_pipeline.py trước."
        )

    # 2. Load model
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"Model không tìm thấy tại {MODEL_PATH}.zip\n"
            "Hãy chạy training_pipeline.py trước."
        )
    model = PPO.load(MODEL_PATH)
    print(f"[Eval] Model loaded từ {MODEL_PATH}.zip")

    # 3. Evaluate
    df = evaluate(test_files, model)

    # 4. Metrics
    metrics = compute_metrics(df)

    # 5. Plots
    plot_solving_times(df, metrics)
    plot_time_distribution(df)

    print("\n[Done] Evaluation hoàn thành!")
