"""
データを各種読み込み、MLが使いやすいptファイルを作る。
＆スケーラーも保存してしまう
1. ファイル名を見てQ-Tの対応を確認する
2. train val test 分割
3. 特徴ベクトル化
4. 正規化　StdScaler
5. ptファイル、pklファイル、jsonなど保存。

ver0の例：/saved/verにしまう

実行例（リポジトリルート）::

python src/preprocess/ml_data_preparation.py \
    --q-patches-dir data/processed/20250825_sample/npz/Q_patches \
    --t-patches-dir data/processed/20250825_sample/npz/T_patches \
    --q-stem heater_out_1 \
    --t-stem sera_temp_1 \
    --output-dir saved/ver0/data_preparated
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# パッチファイル名: {stem}_x{xxxx}_y{yyyy}_rot{deg}_flip{orig|lr}.npz （npz2augmentation.py）
_PATCH_NAME_RE = re.compile(
    r"^(.+)_x(\d+)_y(\d+)_rot(\d+)_flip(\w+)\.npz$"
)


def parse_patch_filename(filename: str) -> Tuple[str, int, int, int, str]:
    """パッチファイル名から stem・座標・オーグメンテーションを抽出。"""
    m = _PATCH_NAME_RE.match(filename)
    if not m:
        raise ValueError(f"想定外のファイル名: {filename}")
    stem, xs, ys, rot, flip = m.groups()
    return stem, int(xs), int(ys), int(rot), flip


def load_patch_data(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    return np.asarray(d["data"])


def list_patch_files(
    patches_dir: Path,
    stem: str,
) -> List[Path]:
    """指定 stem で始まるパッチ NPZ を列挙。"""
    if not patches_dir.is_dir():
        raise FileNotFoundError(f"ディレクトリがありません: {patches_dir}")
    out: List[Path] = []
    for p in sorted(patches_dir.glob("*.npz")):
        try:
            s, _, _, _, _ = parse_patch_filename(p.name)
        except ValueError:
            continue
        if s == stem:
            out.append(p)
    return out


def split_data_by_coordinates(
    q_files: Sequence[Path],
    t_files: Sequence[Path],
    *,
    val_bounds: Tuple[int, int, int, int],
    test_bounds: Tuple[int, int, int, int],
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    (x, y, rot, flip) が一致する Q-T ペアを作り、座標 (x,y) で train/val/test に分割。

    val_bounds / test_bounds: (x_min, x_max, y_min, y_max) いずれも **包含**。
    val と test が重なる場合は ValueError。
    """
    def in_bounds(x: int, y: int, b: Tuple[int, int, int, int]) -> bool:
        xmin, xmax, ymin, ymax = b
        return xmin <= x <= xmax and ymin <= y <= ymax

    q_map: Dict[Tuple[int, int, int, str], Path] = {}
    for q in q_files:
        _, x, y, rot, flip = parse_patch_filename(q.name)
        key = (x, y, rot, flip)
        if key in q_map:
            raise ValueError(f"Q パッチのキー重複: {key} ({q_map[key]} と {q})")
        q_map[key] = q

    t_map: Dict[Tuple[int, int, int, str], Path] = {}
    for t in t_files:
        _, x, y, rot, flip = parse_patch_filename(t.name)
        key = (x, y, rot, flip)
        if key in t_map:
            raise ValueError(f"T パッチのキー重複: {key}")
        t_map[key] = t

    common = set(q_map.keys()) & set(t_map.keys())
    missing_q = set(t_map.keys()) - set(q_map.keys())
    missing_t = set(q_map.keys()) - set(t_map.keys())
    if missing_q or missing_t:
        raise ValueError(
            f"Q/T のキーが一致しません。共通: {len(common)}, Q のみ: {len(set(q_map)-common)}, T のみ: {len(set(t_map)-common)}"
        )

    train_pairs: List[Tuple[Path, Path]] = []
    val_pairs: List[Tuple[Path, Path]] = []
    test_pairs: List[Tuple[Path, Path]] = []

    for key in sorted(common):
        x, y, rot, flip = key
        q_path = q_map[key]
        t_path = t_map[key]
        in_val = in_bounds(x, y, val_bounds)
        in_test = in_bounds(x, y, test_bounds)
        if in_val and in_test:
            raise ValueError(f"(x,y)=({x},{y}) が val と test の両方に該当します。境界を見直してください。")
        if in_val:
            val_pairs.append((q_path, t_path))
        elif in_test:
            test_pairs.append((q_path, t_path))
        else:
            train_pairs.append((q_path, t_path))

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def load_and_preprocess_data(
    pairs: Sequence[Tuple[Path, Path]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """パッチペアを読み込み平坦化。戻り値の dict はチャネル次元情報用。"""
    q_rows: List[np.ndarray] = []
    t_rows: List[np.ndarray] = []
    q_ch0_size: int | None = None
    q_ch1_size: int | None = None
    t_flat_size: int | None = None

    for q_file, t_file in pairs:
        q_patch = load_patch_data(q_file)
        t_patch = load_patch_data(t_file)

        if q_patch.ndim == 3:
            c0 = np.nan_to_num(q_patch[0].ravel(), nan=0.0)
            c1 = np.nan_to_num(q_patch[1].ravel(), nan=0.0)
            q_flat = np.concatenate([c0, c1])
            if q_ch0_size is None:
                q_ch0_size = c0.size
                q_ch1_size = c1.size
        else:
            q_flat = np.nan_to_num(q_patch.ravel(), nan=0.0)
            if q_ch0_size is None:
                q_ch0_size = q_flat.size
                q_ch1_size = 0

        t_flat = np.nan_to_num(t_patch.ravel(), nan=0.0)
        if t_flat_size is None:
            t_flat_size = t_flat.size

        q_rows.append(q_flat)
        t_rows.append(t_flat)

    meta = {
        "q_ch0_elements": q_ch0_size or 0,
        "q_ch1_elements": q_ch1_size or 0,
        "t_elements": t_flat_size or 0,
    }
    if not q_rows:
        return np.zeros((0, 0)), np.zeros((0, 0)), meta
    return np.stack(q_rows, axis=0), np.stack(t_rows, axis=0), meta


def create_ml_dataset(
    q_patches_dir: Path,
    t_patches_dir: Path,
    output_dir: Path,
    *,
    q_stem: str,
    t_stem: str,
    val_bounds: Tuple[int, int, int, int],
    test_bounds: Tuple[int, int, int, int],
) -> None:
    """機械学習用データセット（.pt / scaler .pkl / dataset_info.json）を保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    q_files = list_patch_files(q_patches_dir, q_stem)
    t_files = list_patch_files(t_patches_dir, t_stem)
    if not q_files:
        raise ValueError(f"Q パッチが見つかりません: {q_patches_dir} stem={q_stem}")
    if not t_files:
        raise ValueError(f"T パッチが見つかりません: {t_patches_dir} stem={t_stem}")

    print("データ分割中...")
    data_splits = split_data_by_coordinates(
        q_files, t_files, val_bounds=val_bounds, test_bounds=test_bounds
    )
    print(f"訓練: {len(data_splits['train'])} ペア")
    print(f"検証: {len(data_splits['val'])} ペア")
    print(f"テスト: {len(data_splits['test'])} ペア")

    print("\nデータ読み込み中...")
    train_q, train_t, meta_train = load_and_preprocess_data(data_splits["train"])
    if train_q.size == 0:
        raise ValueError("訓練ペアが 0 です。境界設定または stem を確認してください。")

    print(f"訓練形状: Q={train_q.shape}, T={train_t.shape}")

    def _load_split(pairs: List[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        if not pairs:
            return (
                np.zeros((0, train_q.shape[1]))
                if train_q.shape[1]
                else np.zeros((0, 0)),
                np.zeros((0, train_t.shape[1]))
                if train_t.shape[1]
                else np.zeros((0, 0)),
            )
        q, t, _ = load_and_preprocess_data(pairs)
        return q, t

    val_q, val_t = _load_split(data_splits["val"])
    test_q, test_t = _load_split(data_splits["test"])

    print("\nStandardScaler（訓練のみ fit）...")
    q_scaler = StandardScaler()
    t_scaler = StandardScaler()
    train_q_scaled = q_scaler.fit_transform(train_q)
    train_t_scaled = t_scaler.fit_transform(train_t)
    val_q_scaled = q_scaler.transform(val_q) if val_q.shape[0] else val_q
    val_t_scaled = t_scaler.transform(val_t) if val_t.shape[0] else val_t
    test_q_scaled = q_scaler.transform(test_q) if test_q.shape[0] else test_q
    test_t_scaled = t_scaler.transform(test_t) if test_t.shape[0] else test_t

    print("torch.tensor に変換中...")
    train_q_tensor = torch.as_tensor(train_q_scaled, dtype=torch.float32)
    train_t_tensor = torch.as_tensor(train_t_scaled, dtype=torch.float32)
    val_q_tensor = torch.as_tensor(val_q_scaled, dtype=torch.float32)
    val_t_tensor = torch.as_tensor(val_t_scaled, dtype=torch.float32)
    test_q_tensor = torch.as_tensor(test_q_scaled, dtype=torch.float32)
    test_t_tensor = torch.as_tensor(test_t_scaled, dtype=torch.float32)

    torch.save({"input": train_q_tensor, "target": train_t_tensor}, output_dir / "train_dataset.pt")
    torch.save({"input": val_q_tensor, "target": val_t_tensor}, output_dir / "val_dataset.pt")
    torch.save({"input": test_q_tensor, "target": test_t_tensor}, output_dir / "test_dataset.pt")

    with open(output_dir / "q_scaler.pkl", "wb") as f:
        pickle.dump(q_scaler, f)
    with open(output_dir / "t_scaler.pkl", "wb") as f:
        pickle.dump(t_scaler, f)

    # パッチ空間サイズ（正方形想定のヒント）
    n_q0 = meta_train["q_ch0_elements"]
    side = int(np.sqrt(n_q0)) if n_q0 > 0 else 0
    patch_shape = [side, side] if side * side == n_q0 else [None, None]

    dataset_info = {
        "train_size": int(train_q_tensor.shape[0]),
        "val_size": int(val_q_tensor.shape[0]),
        "test_size": int(test_q_tensor.shape[0]),
        "input_dim": int(train_q_tensor.shape[1]),
        "output_dim": int(train_t_tensor.shape[1]),
        "patch_shape": patch_shape,
        "q_stem": q_stem,
        "t_stem": t_stem,
        "q_channels_flat": {
            "q_value_elements": meta_train["q_ch0_elements"],
            "filter_elements": meta_train["q_ch1_elements"],
        },
        "t_flat_elements": meta_train["t_elements"],
        "val_bounds": list(val_bounds),
        "test_bounds": list(test_bounds),
        "q_scaler_mean": q_scaler.mean_.tolist(),
        "q_scaler_scale": q_scaler.scale_.tolist(),
        "t_scaler_mean": t_scaler.mean_.tolist(),
        "t_scaler_scale": t_scaler.scale_.tolist(),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"\n=== 完了 === 出力: {output_dir}")
    print(f"入力次元: {train_q_tensor.shape[1]}, 出力次元: {train_t_tensor.shape[1]}")


def _parse_bounds(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("x_min x_max y_min y_max の4整数で指定してください。")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q/T パッチ NPZ から train/val/test の .pt と StandardScaler を保存する。"
    )
    parser.add_argument("--q-patches-dir", type=Path, required=True)
    parser.add_argument("--t-patches-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="例: saved/ver0/ml_data")
    parser.add_argument(
        "--q-stem",
        required=True,
        help="Q 側ファイル名の stem（例: heater_out_1 → heater_out_1_x0000_y0000_....npz）",
    )
    parser.add_argument(
        "--t-stem",
        required=True,
        help="T 側ファイル名の stem（例: sera_temp_1）",
    )
    parser.add_argument(
        "--val-bounds",
        type=_parse_bounds,
        default="0 36 0 18",
        metavar="X_MIN X_MAX Y_MIN Y_MAX",
        help="検証に回すウィンドウ左上インデックス (x,y) の範囲（両端含む）。既定: 0 36 0 18",
    )
    parser.add_argument(
        "--test-bounds",
        type=_parse_bounds,
        default="45 81 0 18",
        metavar="X_MIN X_MAX Y_MIN Y_MAX",
        help="テストに回す範囲（両端含む）。既定: 45 81 0 18",
    )
    args = parser.parse_args()

    create_ml_dataset(
        args.q_patches_dir.resolve(),
        args.t_patches_dir.resolve(),
        args.output_dir.resolve(),
        q_stem=args.q_stem,
        t_stem=args.t_stem,
        val_bounds=tuple(args.val_bounds),
        test_bounds=tuple(args.test_bounds),
    )


if __name__ == "__main__":
    main()
