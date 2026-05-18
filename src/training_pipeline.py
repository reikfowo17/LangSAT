import os
import sys
import glob
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from smartsat_env import SmartSATEnv

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception:
    MaskablePPO = None
    ActionMasker = None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IS_KAGGLE = os.path.exists("/kaggle/working")

DATA_DIR   = os.environ.get("LANGSAT_DATA_DIR",
    "/kaggle/input/datasets/heon29/uf20-91" if _IS_KAGGLE else os.path.join(_ROOT, "data"))
OUTPUT_DIR = os.environ.get("LANGSAT_OUTPUT_DIR",
    "/kaggle/working/results" if _IS_KAGGLE else os.path.join(_ROOT, "results"))
MODEL_PATH = os.environ.get("LANGSAT_MODEL_PATH", os.path.join(OUTPUT_DIR, "smartsat_model"))

LEARNING_RATE  = 0.0002
TOTAL_STEPS    = int(os.environ.get("LANGSAT_TOTAL_STEPS", "100000"))  # 1 epoch theo bài báo
TRAIN_RATIO    = 0.8              # 800 train / 200 test
SEED           = 42
SPLIT_SEED     = int(os.environ.get("LANGSAT_SPLIT_SEED", str(SEED)))
CHECKPOINT_FREQ = 10_000          # Lưu checkpoint mỗi 10k steps
TRAIN_LOG_INTERVAL = int(os.environ.get("LANGSAT_TRAIN_LOG_INTERVAL", "5000"))
TRAIN_HEARTBEAT_SECONDS = float(os.environ.get("LANGSAT_TRAIN_HEARTBEAT_SECONDS", "60"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def split_strategy() -> str:
    strategy = os.environ.get("LANGSAT_SPLIT_STRATEGY", "sorted").strip().lower()
    if strategy not in {"sorted", "shuffled"}:
        raise ValueError("LANGSAT_SPLIT_STRATEGY must be 'sorted' or 'shuffled'")
    return strategy


def set_reproducible_seeds(seed: int = SEED):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def load_and_split_dataset(data_dir: str, train_ratio: float = 0.8):
    files = sorted(glob.glob(os.path.join(data_dir, "*.cnf")))

    if not files:
        # Fallback: thử các subdirectory phổ biến
        for sub in ["uf20-91", "uf20", ""]:
            pattern = os.path.join(data_dir, sub, "*.cnf")
            files = sorted(glob.glob(pattern))
            if files:
                break

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file .cnf trong {data_dir}\n"
            f"Hãy download uf20-91 từ https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html\n"
            f"và upload lên Kaggle dưới dạng dataset."
        )

    strategy = split_strategy()
    if strategy == "shuffled":
        rng = np.random.default_rng(SPLIT_SEED)
        files = list(rng.permutation(files))

    n_train = int(len(files) * train_ratio)
    train_files = files[:n_train]
    test_files  = files[n_train:]

    print(f"[Dataset] Tổng: {len(files)} files")
    print(f"[Dataset] Train: {len(train_files)} | Test: {len(test_files)}")
    print(f"[Dataset] Split: {strategy} | Seed: {SPLIT_SEED if strategy == 'shuffled' else 'n/a'}")

    # Lưu split list
    split_info = {
        "train": train_files,
        "test": test_files,
        "metadata": {
            "run_profile": "paper_like",
            "train_ratio": train_ratio,
            "split_strategy": strategy,
            "split_seed": SPLIT_SEED if strategy == "shuffled" else None,
            "n_files": len(files),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "data_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    return train_files, test_files

class RewardLoggerCallback(BaseCallback):
    def __init__(
        self,
        log_interval: int = 1000,
        heartbeat_seconds: float = 60.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.heartbeat_seconds = heartbeat_seconds
        self.rewards = []
        self.steps_log = []
        self._ep_rewards = []
        self._last_logged_step = 0
        self._last_heartbeat_time = time.time()

    def _on_step(self) -> bool:
        # Lấy reward từ infos
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info:
                self._ep_rewards.append(ep_info["r"])

        should_log_step = self.num_timesteps - self._last_logged_step >= self.log_interval
        if should_log_step and self._ep_rewards:
            self._last_logged_step = self.num_timesteps
            self._record_reward()

        now = time.time()
        should_heartbeat = now - self._last_heartbeat_time >= self.heartbeat_seconds
        if should_heartbeat:
            self._last_heartbeat_time = now
            if self.verbose:
                if self._ep_rewards:
                    mean_r = np.mean(self._ep_rewards[-50:])
                    print(
                        f"  Step {self.num_timesteps:>7} | Mean Reward (last 50 ep): {mean_r:.2f}",
                        flush=True,
                    )
                else:
                    print(f"  Step {self.num_timesteps:>7} | waiting for completed episodes", flush=True)

        return True

    def _record_reward(self):
        mean_r = np.mean(self._ep_rewards[-50:])   # mean 50 episodes gần nhất
        self.rewards.append(mean_r)
        self.steps_log.append(self.num_timesteps)
        if self.verbose:
            print(f"  Step {self.num_timesteps:>7} | Mean Reward (last 50 ep): {mean_r:.2f}", flush=True)

    def save_log(self, path: str):
        log = {"steps": self.steps_log, "mean_rewards": self.rewards}
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"[Log] Reward log saved → {path}")

def train_smartsat(train_files: list[str]) -> tuple:
    set_reproducible_seeds(SEED)
    print(" TRAINING SmartSAT")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  Train files   : {len(train_files)}")
    print("  Run profile   : paper_like")
    print()

    def make_env():
        env = SmartSATEnv(train_files)
        env = Monitor(env)
        if ActionMasker is not None:
            env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env

    # Vectorized environment (1 env đủ cho bài báo)
    vec_env = make_vec_env(make_env, n_envs=1, seed=SEED)

    # PPO model; prefer action masking when sb3-contrib is available.
    model_cls = MaskablePPO if MaskablePPO is not None else PPO
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=SEED,
        tensorboard_log=os.path.join(OUTPUT_DIR, "tb_logs"),
    )
    if model_cls is MaskablePPO:
        print("[Train] Using MaskablePPO action masking.")
        model = model_cls(**model_kwargs)
    else:
        print("[Train] sb3-contrib unavailable; falling back to PPO without masking.")
        model = model_cls(**model_kwargs)

    # Callbacks
    reward_callback = RewardLoggerCallback(
        log_interval=TRAIN_LOG_INTERVAL,
        heartbeat_seconds=TRAIN_HEARTBEAT_SECONDS,
        verbose=1,
    )

    print(f"[Train] Bắt đầu training...", flush=True)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=reward_callback,
            progress_bar=False,
        )
    finally:
        vec_env.close()

    elapsed = time.time() - start_time
    print(f"\n[Train] Hoàn thành sau {elapsed/60:.1f} phút")

    # Lưu model
    model.save(MODEL_PATH)
    print(f"[Train] Model saved → {MODEL_PATH}.zip")

    # Lưu reward log
    reward_callback.save_log(os.path.join(OUTPUT_DIR, "training_rewards.json"))

    return model, reward_callback

def plot_reward_curve(callback: RewardLoggerCallback):
    if not callback.steps_log:
        print("[Plot] Không có dữ liệu reward để vẽ.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(callback.steps_log, callback.rewards, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward (last 50 episodes)")
    ax.set_title("SmartSAT Training Reward Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "training_reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Plot] Reward curve saved → {path}")

if __name__ == "__main__":
    # 1. Load dataset
    train_files, test_files = load_and_split_dataset(DATA_DIR, TRAIN_RATIO)

    # 2. Train
    model, callback = train_smartsat(train_files)

    # 3. Plot reward curve
    plot_reward_curve(callback)

    print("\n[Done] Training pipeline hoàn thành!")
    print(f"  Model: {MODEL_PATH}.zip")
    print(f"  Results: {OUTPUT_DIR}/")
