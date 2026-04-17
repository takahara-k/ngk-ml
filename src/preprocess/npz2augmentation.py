"""
フルグリッド NPZ（csv2npz.py の出力）を読み、スライディングウィンドウ＋Augmentation でパッチ NPZ を生成する。

- Q: data shape (2, H, W)
- T: data shape (H, W)

Augmentation: rot90 を k=0,1,2,3 と fliplr の有無（計 8 パターン／窓あたり）。

実行例（リポジトリルートで。先に csv2npz でフル NPZ を input-dir に置く）::

python src/preprocess/npz2augmentation.py \
    --input-dir data/processed/20250825_sample/npz \
    --window-size 20 20 \
    --stride 9

パッチは ``input-dir/Q_patches`` と ``input-dir/T_patches`` に保存する（NPZ の kind または形状で振り分け）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _npz_meta_str(z: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in z.files:
        return None
    arr = z[key]
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def load_full_grid(npz_path: Path) -> tuple[np.ndarray, dict[str, str | None]]:
    """csv2npz 形式の NPZ を読み、data とメタを返す。"""
    z = np.load(npz_path)
    if "data" not in z.files:
        raise ValueError(f"{npz_path}: 'data' がありません。keys={z.files}")
    grid = z["data"]
    meta: dict[str, str | None] = {
        "kind": _npz_meta_str(z, "kind"),
        "value_column": _npz_meta_str(z, "value_column"),
    }
    return grid, meta


def patch_subdir_for_npz(meta: dict[str, str | None], grid: np.ndarray) -> str:
    """input-dir 直下に作るサブフォルダ名（Q_patches / T_patches）。"""
    kind = meta.get("kind")
    if kind == "q":
        return "Q_patches"
    if kind == "t":
        return "T_patches"
    if grid.ndim == 3:
        return "Q_patches"
    if grid.ndim == 2:
        return "T_patches"
    raise ValueError(f"Q/T を判別できません: kind={kind}, data.shape={grid.shape}")


def create_sliding_windows_from_npz(
    npz_path: Path,
    window_size: tuple[int, int],
    stride: int,
    output_dir: Path,
    prefix: str | None = None,
    *,
    grid: np.ndarray | None = None,
    meta: dict[str, str | None] | None = None,
) -> list[Path]:
    """
    1 つのフル NPZ からスライディングウィンドウでパッチを切り、Augmentation 付きで保存する。

    window_size: (height, width) — グリッドの第0軸・第1軸に対応。
    grid / meta を省略すると npz_path から読み込む。
    """
    if grid is None or meta is None:
        grid, meta = load_full_grid(npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if grid.ndim == 2:
        height, width = grid.shape
        is_multichannel = False
    elif grid.ndim == 3:
        _, height, width = grid.shape
        is_multichannel = True
    else:
        raise ValueError(f"想定しない data の次元: {grid.shape}")

    window_h, window_w = window_size
    if window_h <= 0 or window_w <= 0:
        raise ValueError("window_size は正の整数にしてください。")
    if height < window_h or width < window_w:
        raise ValueError(
            f"窓がグリッドより大きいです: grid={height}x{width}, window={window_h}x{window_w}"
        )

    stem = prefix if prefix is not None else npz_path.stem
    all_paths: list[Path] = []

    for y in range(0, height - window_h + 1, stride):
        for x in range(0, width - window_w + 1, stride):
            if is_multichannel:
                patch = grid[:, y : y + window_h, x : x + window_w]
            else:
                patch = grid[y : y + window_h, x : x + window_w]
            for k in [0, 1, 2, 3]:
                if is_multichannel:
                    rotated = np.zeros_like(patch)
                    for c in range(patch.shape[0]):
                        rotated[c] = np.rot90(patch[c], k=k)
                else:
                    rotated = np.rot90(patch, k=k)
                deg = (k * 90) % 360
                for flip in [False, True]:
                    if is_multichannel:
                        arr = np.zeros_like(rotated)
                        for c in range(rotated.shape[0]):
                            arr[c] = np.fliplr(rotated[c]) if flip else rotated[c]
                    else:
                        arr = np.fliplr(rotated) if flip else rotated
                    flip_tag = "lr" if flip else "orig"
                    fname = f"{stem}_x{x:04d}_y{y:04d}_rot{deg}_flip{flip_tag}.npz"
                    fpath = output_dir / fname
                    save_kw: dict = {
                        "data": arr,
                        "x_start": x,
                        "y_start": y,
                        "rot_deg": deg,
                        "flip": flip_tag,
                        "source_npz": npz_path.name,
                    }
                    if meta.get("kind") is not None:
                        save_kw["kind"] = meta["kind"]
                    if meta.get("value_column") is not None:
                        save_kw["value_column"] = meta["value_column"]
                    np.savez(fpath, **save_kw)
                    all_paths.append(fpath)

    return all_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フルグリッド NPZ をスライディングウィンドウ＋Augmentation でパッチ NPZ に分割する。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="フル NPZ が入っているディレクトリ（直下の *.npz のみ）。その配下に Q_patches/ T_patches/ を作成して保存する。",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        required=True,
        metavar=("H", "W"),
        help="パッチの高さ・幅（格子セル数）。例: --window-size 20 20",
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="スライドのストライド（ピクセル）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="出力ファイル名の接頭辞（省略時は各入力 NPZ の stem を使用）",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir がディレクトリではありません: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"警告: NPZ が見つかりません: {input_dir}")
        return

    total = 0
    for npz_path in npz_files:
        grid, meta = load_full_grid(npz_path)
        subdir = patch_subdir_for_npz(meta, grid)
        patch_dir = input_dir / subdir
        prefix = args.prefix if args.prefix else npz_path.stem
        paths = create_sliding_windows_from_npz(
            npz_path,
            (args.window_size[0], args.window_size[1]),
            args.stride,
            patch_dir,
            prefix=prefix,
            grid=grid,
            meta=meta,
        )
        total += len(paths)
        print(f"{npz_path.name}: {len(paths)} ファイル -> {patch_dir}")

    print(f"合計 {total} パッチ NPZ を保存しました。")


if __name__ == "__main__":
    main()
