"""
save_results/ 以下の *_final_results.txt を集計して平均・標準偏差を表示するスクリプト。
"""

import re
from pathlib import Path

import numpy as np


METRICS = [
    ("train_loss", r"Train Loss:\s*([\d.]+)"),
    ("test_loss", r"Test_loss:\s*([\d.]+)"),
    ("train_acc", r"Train Acc:\s*([\d.]+)"),
    ("test_acc", r"Test Acc:\s*([\d.]+)"),
    ("best_clients_avg_acc", r"Best Clients AVG Acc:\s*([\d.]+)"),
    ("best_global_model_acc", r"Best Global Model Acc:\s*([\d.]+)"),
]

# 収束の定義: 最終到達精度の CONVERGENCE_THRESHOLD 倍に初めて達したラウンド
CONVERGENCE_THRESHOLD = 0.90


def convergence_round(acc_per_round: list[float], threshold: float = CONVERGENCE_THRESHOLD) -> int | None:
    """acc_per_round の最大値の threshold 倍に初めて達したラウンド番号(1始まり)を返す。"""
    if not acc_per_round:
        return None
    target = max(acc_per_round) * threshold
    for i, acc in enumerate(acc_per_round):
        if acc >= target:
            return i + 1
    return None


def parse_result_file(path: Path) -> dict[str, float | list[float]] | None:
    text = path.read_text()
    result: dict[str, float | list[float]] = {}
    for key, pattern in METRICS:
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))
    m = re.search(r"Acc Per Round:\s*([\d.,]+)", text)
    if m:
        result["acc_per_round"] = [float(v) for v in m.group(1).split(",") if v]
    return result if result else None


def summarize_directory(result_dir: Path) -> None:
    files = sorted(result_dir.glob("*_final_results.txt"))
    if not files:
        return

    file_record_pairs: list[tuple[Path, dict[str, float]]] = []
    for f in files:
        parsed = parse_result_file(f)
        if parsed:
            file_record_pairs.append((f, parsed))
    records = [r for _, r in file_record_pairs]

    if not records:
        print(f"  結果ファイルが見つかりません: {result_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  {result_dir.relative_to(Path('save_results'))}")
    print(f"  試行数: {len(records)}")
    print(f"{'=' * 60}")

    # 各テストケースの結果
    active_keys = [key for key, _ in METRICS if any(key in r for r in records)]
    header = f"  {'Trial':<8}" + "".join(f"{k:>25}" for k in active_keys)
    print(header)
    print(f"  {'-' * (8 + 25 * len(active_keys))}")
    for i, (f, r) in enumerate(file_record_pairs, 1):
        row = f"  {f.stem:<8}" + "".join(
            f"{r[k]:>25.4f}" if k in r else f"{'N/A':>25}" for k in active_keys
        )
        print(row)

    # 集計
    print(f"  {'-' * (8 + 25 * len(active_keys))}")
    print(f"\n  {'Metric':<25} {'Mean':>10} {'Std':>10}")
    print(f"  {'-' * 45}")

    for key, _ in METRICS:
        values = [r[key] for r in records if key in r]
        if not values:
            continue
        arr = np.array(values)
        print(f"  {key:<25} {arr.mean():>10.4f} {arr.std():>10.4f}")

    # 収束ラウンド
    conv_rounds = [
        convergence_round(r["acc_per_round"])
        for r in records
        if "acc_per_round" in r
    ]
    conv_rounds = [c for c in conv_rounds if c is not None]
    if conv_rounds:
        arr = np.array(conv_rounds, dtype=float)
        print(f"\n  収束ラウンド ({int(CONVERGENCE_THRESHOLD*100)}% of max acc)")
        print(f"  {'-' * 45}")
        print(f"  {'convergence_round':<25} {arr.mean():>10.1f} {arr.std():>10.1f}")
        for i, (f, r) in enumerate(file_record_pairs, 1):
            if "acc_per_round" in r:
                cr = convergence_round(r["acc_per_round"])
                print(f"    {f.stem}: round {cr} / {len(r['acc_per_round'])}")

    print()


def main() -> None:
    save_dir = Path("save_results")
    if not save_dir.exists():
        print(f"ディレクトリが見つかりません: {save_dir}")
        return

    # *_final_results.txt を含むディレクトリを探索
    dirs_with_results = sorted(
        {f.parent for f in save_dir.rglob("*_final_results.txt")}
    )

    if not dirs_with_results:
        print("結果ファイルが見つかりませんでした。")
        return

    print(f"結果ディレクトリ数: {len(dirs_with_results)}")
    for d in dirs_with_results:
        summarize_directory(d)


if __name__ == "__main__":
    main()
