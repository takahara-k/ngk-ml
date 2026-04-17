"""
CSVをNPZに変換するだけ：
- input_dir: csv
- output_dir: npz

main関数内で、argparseで、input_dir, output_dirで実行できるようにする。

実行例（リポジトリルートで）::
python src/preprocess/csv2npz.py \
    --input-dir data/processed/20250825_sample \
    --output-dir data/processed/20250825_sample/npz

入力 CSV と同じファイル名（拡張子のみ .npz）が output-dir に保存される。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

# 発熱 / 温度の値列（先にマッチしたものを採用）
_Q_COLUMNS = ("Q_1", "Q_2", "heater_output_W_m-3")
_T_COLUMNS = ("T_1", "T_2", "temperature_degC")


def _build_xy_grid(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    agg = (
        df.groupby(["Y", "X"], as_index=False)[value_col]
        .mean()
        .sort_values(["Y", "X"])
    )
    xs = np.unique(agg["X"].to_numpy())
    ys = np.unique(agg["Y"].to_numpy())
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for _, row in agg.iterrows():
        grid[yi[row["Y"]], xi[row["X"]]] = row[value_col]
    return xs, ys, grid


def _edges(vals: np.ndarray) -> np.ndarray:
    if vals.size == 1:
        v = vals[0]
        return np.array([v - 0.5, v + 0.5])
    diffs = np.diff(vals)
    step = np.median(diffs)
    start = vals[0] - step / 2
    end = vals[-1] + step / 2
    return np.linspace(start, end, vals.size + 1)


def create_q_filter(q_grid: np.ndarray) -> np.ndarray:
    """Qグリッドからフィルターを作成（有限値=1、非有限=0、四辺=-1）。"""
    filter_grid = np.where(np.isfinite(q_grid), 1.0, 0.0)
    filter_grid[0, :] = -1.0
    filter_grid[-1, :] = -1.0
    filter_grid[:, 0] = -1.0
    filter_grid[:, -1] = -1.0
    return filter_grid


def _detect_kind_and_column(df: pd.DataFrame) -> tuple[str, Literal["q", "t"]]:
    cols = set(df.columns)
    for c in _Q_COLUMNS:
        if c in cols:
            return c, "q"
    for c in _T_COLUMNS:
        if c in cols:
            return c, "t"
    raise ValueError(
        f"想定する値列が見つかりません。Q: {_Q_COLUMNS} / T: {_T_COLUMNS}。実際の列: {list(df.columns)}"
    )


def _read_prepared_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: 必須列がありません {sorted(missing)}")
    return df.dropna(subset=["X", "Y", "Z"]).copy()


def csv_to_npz(csv_path: Path, output_dir: Path) -> Path:
    """
    1本のCSVをフルグリッドNPZに保存する。
    - Q系: data shape (2, H, W) — [Q値, フィルタ]
    - T系: data shape (H, W)
    保存キー: data, x_coords, y_coords, x_edges, y_edges
    ファイル名: 入力と同じstem + .npz
    """
    df = _read_prepared_df(csv_path)
    value_col, kind = _detect_kind_and_column(df)
    xs, ys, grid = _build_xy_grid(df, value_col)
    x_edges = _edges(xs)
    y_edges = _edges(ys)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}.npz"

    if kind == "q":
        q_filter = create_q_filter(grid)
        q_2ch = np.stack([grid, q_filter], axis=0)
        np.savez(
            out_path,
            data=q_2ch,
            x_coords=xs,
            y_coords=ys,
            x_edges=x_edges,
            y_edges=y_edges,
            value_column=value_col,
            kind="q",
        )
    else:
        np.savez(
            out_path,
            data=grid,
            x_coords=xs,
            y_coords=ys,
            x_edges=x_edges,
            y_edges=y_edges,
            value_column=value_col,
            kind="t",
        )
    return out_path


def convert_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """input_dir 直下の *.csv をすべて NPZ に変換し、書き出したパスのリストを返す。"""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        return []
    written: list[Path] = []
    for csv_path in csv_files:
        written.append(csv_to_npz(csv_path, output_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="CSVディレクトリ内のCSVを同名のNPZに変換する（フルグリッドのみ）。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="CSVが入っているディレクトリ（直下の *.csv のみ対象）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="NPZの出力先ディレクトリ",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"input-dir がディレクトリではありません: {input_dir}")

    written = convert_directory(input_dir, output_dir)
    if not written:
        print(f"警告: CSV が見つかりませんでした: {input_dir}")
        return

    for p in written:
        print("Saved:", p)


if __name__ == "__main__":
    main()
