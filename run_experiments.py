"""
CIFAR-10 を使って FLIS-HC を 5 回実行するスクリプト。
各試行でシードを変えて再現性のある結果を収集する。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# 実験設定
BASE_CONFIG = {
    "rounds": 500,
    "num_users": 100,
    "frac": 0.1,
    "local_ep": 10,
    "local_bs": 10,
    "lr": 0.01,
    "momentum": 0.5,
    "model": "lenet5",
    "dataset": "cifar10",
    "datadir": "./data/",
    "savedir": "./save_results/",
    "logdir": "./logs/",
    "partition": "noniid-#label2",
    "alg": "flis_hc",
    "beta": 0.1,
    "noise": 0,
    "cluster_alpha": 0.45,
    "nclasses": 10,
    "nsamples_shared": 2500,
    "gpu": 0,
    "print_freq": 50,
}

NUM_TRIALS = 5
SEEDS = [1, 2, 3, 4, 5]


def build_command(trial: int, seed: int) -> list[str]:
    cmd = [sys.executable, "main_FLIS_HC.py"]
    cmd += [f"--trial={trial}"]
    cmd += [f"--seed={seed}"]
    cmd += ["--local_view"]
    for key, value in BASE_CONFIG.items():
        cmd.append(f"--{key}={value}")
    return cmd


def run_trial(trial: int, seed: int, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"trial_{trial}_seed_{seed}.txt"

    cmd = build_command(trial, seed)
    print(f"\n{'=' * 60}")
    print(f"Trial {trial}/{NUM_TRIALS}  (seed={seed})")
    print(f"Log: {log_file}")
    print(f"{'=' * 60}")

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(result.stdout)

    # 最終行付近を表示して進捗確認
    lines = result.stdout.strip().splitlines()
    for line in lines[-10:]:
        print(line)

    return result.returncode


def main():
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    log_dir = Path("logs/experiments")
    results = []

    for i, seed in enumerate(SEEDS, start=1):
        returncode = run_trial(trial=i, seed=seed, log_dir=log_dir)
        results.append({"trial": i, "seed": seed, "returncode": returncode})
        status = "OK" if returncode == 0 else f"FAILED (code={returncode})"
        print(f"Trial {i} finished: {status}")

    print(f"\n{'=' * 60}")
    print("全試行完了")
    print(f"{'=' * 60}")
    for r in results:
        status = "OK" if r["returncode"] == 0 else f"FAILED (code={r['returncode']})"
        print(f"  Trial {r['trial']} (seed={r['seed']}): {status}")

    # サマリを JSON で保存
    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nサマリ保存先: {summary_path}")


if __name__ == "__main__":
    main()
