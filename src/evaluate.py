import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from cdcl_baseline import CDCLSolver, PYSAT_TIME_MODE, SATInstance
from satfeat_adapter import (
    BACKEND_USAGE,
    FEATURE_BACKEND,
    SATFEATPY_DIR,
    SATFEATPY_FULL_LOCAL_SEARCH,
    extract_sat_features,
)
from smartsat_env import N_VARS, build_solver_observation

try:
    from sb3_contrib import MaskablePPO
except Exception:
    MaskablePPO = None


# Đường dẫn: tự động phát hiện local vs Kaggle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IS_KAGGLE = os.path.exists("/kaggle/working")

OUTPUT_DIR = os.environ.get("LANGSAT_OUTPUT_DIR",
    "/kaggle/working/results" if _IS_KAGGLE else os.path.join(_ROOT, "results"))
MODEL_PATH = os.environ.get("LANGSAT_MODEL_PATH", os.path.join(OUTPUT_DIR, "smartsat_model"))
SPLIT_PATH = os.environ.get("LANGSAT_SPLIT_PATH", os.path.join(OUTPUT_DIR, "data_split.json"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAPER_MEDIAN_SECONDS = 1.02
TIME_SCALE = float(os.environ.get("LANGSAT_TIME_SCALE", "1.0"))
REPORT_SCALE_TO_PAPER = os.environ.get("LANGSAT_REPORT_SCALE_TO_PAPER", "0") == "1"
SMARTSAT_POLICY_MODE = os.environ.get("LANGSAT_POLICY_MODE", "rl").lower()
SMARTSAT_USE_SEARCH_TIME = os.environ.get("LANGSAT_USE_SEARCH_TIME", "0") == "1"
BASELINE_USE_PYSAT = os.environ.get("LANGSAT_BASELINE_USE_PYSAT", os.environ.get("LANGSAT_USE_PYSAT", "1")) == "1"
SMARTSAT_USE_PYSAT = os.environ.get("LANGSAT_SMARTSAT_USE_PYSAT", os.environ.get("LANGSAT_USE_PYSAT", "1")) == "1"
PAPERLIKE_PYSAT_TIME_MODE = os.environ.get("LANGSAT_PYSAT_TIME_MODE", PYSAT_TIME_MODE).lower()


def _time_scale(df: pd.DataFrame) -> float:
    if TIME_SCALE != 1.0:
        return TIME_SCALE
    if not REPORT_SCALE_TO_PAPER:
        return 1.0
    medians = [df["baseline_time_raw"].median(), df["smartsat_time_raw"].median()]
    positive = [m for m in medians if m and m > 0]
    if not positive:
        return 1.0
    return PAPER_MEDIAN_SECONDS / float(np.mean(positive))


def _predict_action(model, obs, action_masks=None):
    if action_masks is not None and MaskablePPO is not None and model.__class__.__name__ == "MaskablePPO":
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        return action
    if action_masks is not None and hasattr(model, "predict"):
        try:
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            return action
        except TypeError:
            pass
    action, _ = model.predict(obs, deterministic=True)
    return action


def solve_with_smartsat(filepath: str, model: PPO) -> tuple[bool, float, dict]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    global_features = extract_sat_features(filepath)
    policy_time = 0.0
    rl_decisions = 0
    fallback_decisions = 0

    def policy(current_solver: CDCLSolver):
        nonlocal policy_time, rl_decisions, fallback_decisions
        baseline_pick = current_solver.pick_branching_variable()
        if baseline_pick is None:
            return None

        obs = build_solver_observation(current_solver, global_features)
        action_masks = None
        if hasattr(current_solver, "assignment"):
            mask = np.zeros(N_VARS * 2, dtype=np.int8)
            for var in range(1, min(current_solver.inst.n_vars, N_VARS) + 1):
                if current_solver.assignment[var] == 0:
                    mask[(var - 1) * 2] = 1
                    mask[(var - 1) * 2 + 1] = 1
            action_masks = mask
        start = time.perf_counter()
        action = _predict_action(model, obs, action_masks)
        policy_time += time.perf_counter() - start
        action = int(action)
        var = action // 2 + 1
        value = 1 if action % 2 == 1 else -1

        if SMARTSAT_POLICY_MODE == "rl":
            if 1 <= var <= current_solver.inst.n_vars and current_solver.assignment[var] == 0:
                rl_decisions += 1
                return var, value
            fallback_decisions += 1
            return baseline_pick

        baseline_var, baseline_value = baseline_pick
        if var == baseline_var and current_solver.assignment[var] == 0:
            rl_decisions += 1
            return var, value
        fallback_decisions += 1
        return baseline_var, baseline_value

    sat, elapsed = solver.solve(
        decision_policy=policy,
        use_pysat_fallback=SMARTSAT_USE_PYSAT,
        pysat_time_mode=PAPERLIKE_PYSAT_TIME_MODE,
    )
    search_elapsed = max(elapsed - policy_time, 0.0)
    return sat, elapsed, {
        "search_time_raw": search_elapsed,
        "policy_time_raw": policy_time,
        "decisions": solver.stats.decisions,
        "propagations": solver.stats.propagations,
        "conflicts": solver.stats.conflicts,
        "policy_calls": solver.stats.policy_calls,
        "learned_clauses": solver.stats.learned_clauses,
        "engine": solver.stats.engine,
        "budget_exceeded": solver.stats.budget_exceeded,
        "timed_out": solver.stats.timed_out,
        "rl_decisions": rl_decisions,
        "fallback_decisions": fallback_decisions,
        "invalid_action_rate": _safe_div(fallback_decisions, solver.stats.policy_calls),
        "policy_time_per_call_raw": _safe_div(policy_time, solver.stats.policy_calls),
    }


def solve_baseline_with_stats(filepath: str) -> tuple[bool, float, dict]:
    inst = SATInstance.from_dimacs(filepath)
    solver = CDCLSolver(inst)
    sat, elapsed = solver.solve(
        use_pysat_fallback=BASELINE_USE_PYSAT,
        pysat_time_mode=PAPERLIKE_PYSAT_TIME_MODE,
    )
    return sat, elapsed, {
        "decisions": solver.stats.decisions,
        "propagations": solver.stats.propagations,
        "conflicts": solver.stats.conflicts,
        "policy_calls": solver.stats.policy_calls,
        "learned_clauses": solver.stats.learned_clauses,
        "engine": solver.stats.engine,
        "budget_exceeded": solver.stats.budget_exceeded,
        "timed_out": solver.stats.timed_out,
    }


def evaluate(test_files: list[str], model: PPO) -> pd.DataFrame:
    results = []
    n = len(test_files)

    print(f" EVALUATION — {n} instances")

    for i, filepath in enumerate(test_files):
        # 1. CDCL Baseline
        baseline_sat, baseline_time, baseline_stats = solve_baseline_with_stats(filepath)

        # 2. SmartSAT
        smartsat_sat, smartsat_time, smartsat_stats = solve_with_smartsat(filepath, model)

        results.append({
            "file": os.path.basename(filepath),
            "instance_idx": i,
            "baseline_sat": baseline_sat,
            "baseline_time_raw": baseline_time,
            "baseline_search_time_raw": baseline_time,
            "baseline_decisions": baseline_stats["decisions"],
            "baseline_propagations": baseline_stats["propagations"],
            "baseline_conflicts": baseline_stats["conflicts"],
            "baseline_learned_clauses": baseline_stats["learned_clauses"],
            "baseline_engine": baseline_stats["engine"],
            "baseline_budget_exceeded": baseline_stats["budget_exceeded"],
            "baseline_timed_out": baseline_stats["timed_out"],
            "smartsat_sat": smartsat_sat,
            "smartsat_time_raw": smartsat_time,
            "smartsat_search_time_raw": smartsat_stats["search_time_raw"],
            "smartsat_policy_time_raw": smartsat_stats["policy_time_raw"],
            "smartsat_decisions": smartsat_stats["decisions"],
            "smartsat_propagations": smartsat_stats["propagations"],
            "smartsat_conflicts": smartsat_stats["conflicts"],
            "smartsat_policy_calls": smartsat_stats["policy_calls"],
            "smartsat_learned_clauses": smartsat_stats["learned_clauses"],
            "smartsat_engine": smartsat_stats["engine"],
            "smartsat_budget_exceeded": smartsat_stats["budget_exceeded"],
            "smartsat_timed_out": smartsat_stats["timed_out"],
            "smartsat_rl_decisions": smartsat_stats["rl_decisions"],
            "smartsat_fallback_decisions": smartsat_stats["fallback_decisions"],
            "smartsat_invalid_action_rate": smartsat_stats["invalid_action_rate"],
            "smartsat_policy_time_per_call_raw": smartsat_stats["policy_time_per_call_raw"],
        })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:>3}/{n}] Baseline: {baseline_time:.4f}s | SmartSAT: {smartsat_time:.4f}s")

    df = pd.DataFrame(results)
    scale = _time_scale(df)
    df["baseline_time"] = df["baseline_time_raw"] * scale
    df["smartsat_time"] = df["smartsat_time_raw"] * scale
    df["baseline_search_time"] = df["baseline_search_time_raw"] * scale
    df["smartsat_search_time"] = df["smartsat_search_time_raw"] * scale
    if SMARTSAT_USE_SEARCH_TIME:
        df["baseline_time"] = df["baseline_search_time"]
        df["smartsat_time"] = df["smartsat_search_time"]
    df["time_scale"] = scale
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

    decision_ratio = _median_ratio(df["smartsat_decisions"], df["baseline_decisions"])
    conflict_ratio = _median_ratio(df["smartsat_conflicts"], df["baseline_conflicts"])

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
        "median_smartsat_raw": round(float(df["smartsat_time_raw"].median()), 6),
        "median_baseline_raw": round(float(df["baseline_time_raw"].median()), 6),
        "median_smartsat_search_raw": round(float(df["smartsat_search_time_raw"].median()), 6),
        "median_baseline_search_raw": round(float(df["baseline_search_time_raw"].median()), 6),
        "median_policy_time_raw": round(float(df["smartsat_policy_time_raw"].median()), 6),
        "median_policy_time_per_call_raw": round(float(df["smartsat_policy_time_per_call_raw"].median()), 8),
        "time_scale"       : round(float(df["time_scale"].iloc[0]), 6) if "time_scale" in df else 1.0,
        "policy_mode"       : SMARTSAT_POLICY_MODE,
        "use_search_time"   : SMARTSAT_USE_SEARCH_TIME,
        "baseline_use_pysat": BASELINE_USE_PYSAT,
        "smartsat_use_pysat": SMARTSAT_USE_PYSAT,
        "pysat_time_mode"   : PAPERLIKE_PYSAT_TIME_MODE,
        "feature_backend"   : FEATURE_BACKEND,
        "feature_backend_usage": dict(BACKEND_USAGE),
        "satfeatpy_dir"     : SATFEATPY_DIR,
        "satfeatpy_full_local_search": SATFEATPY_FULL_LOCAL_SEARCH,
        "sat_rate_smartsat": round(df["smartsat_sat"].mean() * 100, 2),
        "sat_rate_baseline": round(df["baseline_sat"].mean() * 100, 2),
        "median_decisions_smartsat": round(float(df["smartsat_decisions"].median()), 2),
        "median_decisions_baseline": round(float(df["baseline_decisions"].median()), 2),
        "median_conflicts_smartsat": round(float(df["smartsat_conflicts"].median()), 2),
        "median_conflicts_baseline": round(float(df["baseline_conflicts"].median()), 2),
        "median_rl_decisions": round(float(df["smartsat_rl_decisions"].median()), 2),
        "median_fallback_decisions": round(float(df["smartsat_fallback_decisions"].median()), 2),
        "median_policy_calls": round(float(df["smartsat_policy_calls"].median()), 2),
        "median_invalid_action_rate_pct": round(float(df["smartsat_invalid_action_rate"].median()) * 100, 2),
        "median_decision_ratio_st_over_bsl": round(decision_ratio, 4),
        "median_conflict_ratio_st_over_bsl": round(conflict_ratio, 4),
        "baseline_budget_exit_rate_pct": round(float(df["baseline_budget_exceeded"].mean()) * 100, 2),
        "smartsat_budget_exit_rate_pct": round(float(df["smartsat_budget_exceeded"].mean()) * 100, 2),
        "baseline_budget_exits": int(df["baseline_budget_exceeded"].sum()),
        "smartsat_budget_exits": int(df["smartsat_budget_exceeded"].sum()),
        "baseline_pysat_fallbacks": int((df["baseline_engine"] == "pysat_minisat22").sum()),
        "smartsat_pysat_fallbacks": int((df["smartsat_engine"] == "pysat_minisat22").sum()),
    }

    # In bảng kết quả
    print("  KẾT QUẢ EVALUATION")
    print(f"  Số instances test    : {metrics['n_instances']}")
    print(f"  SmartSAT thắng       : {metrics['smartsat_wins']} ({metrics['win_rate_pct']}%)")
    print(f"  Baseline thắng       : {metrics['baseline_wins']}")
    print(f"  Hòa                  : {metrics['ties']}")
    print(f"  Median SmartSAT      : {metrics['median_smartsat']}s")
    print(f"  Median Baseline      : {metrics['median_baseline']}s")
    print(f"  Policy mode          : {metrics['policy_mode']}")
    print(f"  Feature backend      : {metrics['feature_backend']}")
    print(f"  Feature usage        : {metrics['feature_backend_usage']}")
    print(f"  Search-time metric   : {metrics['use_search_time']}")
    print(f"  PySAT time mode      : {metrics['pysat_time_mode']}")
    print(f"  Baseline budget exits: {metrics['baseline_budget_exits']}")
    print(f"  SmartSAT budget exits: {metrics['smartsat_budget_exits']}")
    print(f"  PySAT fallbacks B/ST : {metrics['baseline_pysat_fallbacks']}/{metrics['smartsat_pysat_fallbacks']}")
    print(f"  Median decisions ST  : {metrics['median_decisions_smartsat']}")
    print(f"  Median decisions BSL : {metrics['median_decisions_baseline']}")
    print(f"  Median policy calls  : {metrics['median_policy_calls']}")
    print(f"  Invalid action rate  : {metrics['median_invalid_action_rate_pct']}%")
    print(f"  Policy time / call   : {metrics['median_policy_time_per_call_raw']}s")
    print(f"  Decision ratio ST/BSL: {metrics['median_decision_ratio_st_over_bsl']}x")
    print(f"  Conflict ratio ST/BSL: {metrics['median_conflict_ratio_st_over_bsl']}x")
    print(f"  [Bài báo gốc]        : ~53% win rate, ~1.02s median")
    print("="*55)

    # So sánh với bài báo
    print(f"  Raw Median SmartSAT  : {metrics['median_smartsat_raw']}s")
    print(f"  Raw Median Baseline  : {metrics['median_baseline_raw']}s")
    print(f"  Raw Search SmartSAT  : {metrics['median_smartsat_search_raw']}s")
    print(f"  Raw Policy Overhead  : {metrics['median_policy_time_raw']}s")
    print(f"  Time scale reported  : {metrics['time_scale']}x")
    print(f"\n  Sai lệch win rate   : {abs(metrics['win_rate_pct'] - 53.0):.2f}%")
    print(f"  Sai lệch median ST  : {abs(metrics['median_smartsat'] - PAPER_MEDIAN_SECONDS):.4f}s")
    print(f"  Sai lệch median BSL : {abs(metrics['median_baseline'] - PAPER_MEDIAN_SECONDS):.4f}s")

    # Lưu metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {OUTPUT_DIR}/metrics.json")

    return metrics


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _median_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    denom = denominator.replace(0, np.nan)
    ratios = (numerator / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return 0.0
    return float(ratios.median())


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
    if MaskablePPO is not None:
        try:
            model = MaskablePPO.load(MODEL_PATH)
            print(f"[Eval] MaskablePPO model loaded từ {MODEL_PATH}.zip")
        except Exception:
            model = PPO.load(MODEL_PATH)
            print(f"[Eval] PPO model loaded từ {MODEL_PATH}.zip")
    else:
        model = PPO.load(MODEL_PATH)
        print(f"[Eval] PPO model loaded từ {MODEL_PATH}.zip")

    # 3. Evaluate
    df = evaluate(test_files, model)

    # 4. Metrics
    metrics = compute_metrics(df)

    # 5. Plots
    plot_solving_times(df, metrics)
    plot_time_distribution(df)

    print("\n[Done] Evaluation hoàn thành!")
