"""
パッチ NPZ（npz2augmentation.py の出力）をグリッド画像として PNG 保存する。

- ``--input-dir`` に ``Q_patches`` / ``T_patches`` を含む親フォルダ、またはパッチフォルダそのものを指定可能。

実行例（リポジトリルート）::

# 親フォルダ（Q_patches / T_patches の両方を viz に出力）
python src/preprocess/npz2png.py --input-dir data/processed/20250825_sample/npz --all-augs --output-dir saved/ver0/patch_visualizations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _scalar_str(d: object, key: str) -> Optional[str]:
    if key not in d.files:
        return None
    a = d[key]
    if a.shape == ():
        return str(a.item())
    return str(a)


def _scalar_int(d: object, key: str) -> Optional[int]:
    if key not in d.files:
        return None
    a = d[key]
    if a.shape == ():
        return int(a.item())
    return int(a)


def collect_npz_paths(npz_dir: Path) -> List[Path]:
    return sorted(npz_dir.glob("*.npz"))


def resolve_patch_directories(input_dir: Path) -> List[Path]:
    """
    input-dir が Q_patches / T_patches を含む親ならそのサブフォルダを列挙。
    それ以外は input-dir 自体（パッチが直下にある想定）。
    """
    qp = input_dir / "Q_patches"
    tp = input_dir / "T_patches"
    out: List[Path] = []
    if qp.is_dir() and any(qp.glob("*.npz")):
        out.append(qp)
    if tp.is_dir() and any(tp.glob("*.npz")):
        out.append(tp)
    if out:
        return out
    return [input_dir]


def detect_two_channel(npz_paths: Sequence[Path]) -> bool:
    if not npz_paths:
        return False
    d = np.load(npz_paths[0])
    arr = d["data"]
    return arr.ndim == 3


def collect_coords(
    npz_paths: Sequence[Path],
    require_aug: Optional[Tuple[int, str]] = None,
) -> List[Tuple[int, int, Path]]:
    coords: List[Tuple[int, int, Path]] = []
    for p in npz_paths:
        d = np.load(p)
        rot = _scalar_int(d, "rot_deg")
        flip = _scalar_str(d, "flip")
        if require_aug is not None:
            req_rot, req_flip = require_aug
            if rot is None or rot != req_rot:
                continue
            if flip is None or flip != req_flip:
                continue
        else:
            if rot is not None and rot != 0:
                continue
            if flip is not None and flip != "orig":
                continue
        xs = int(_scalar_int(d, "x_start") or 0)
        ys = int(_scalar_int(d, "y_start") or 0)
        coords.append((xs, ys, p))
    return coords


def visualize_npz_patches(
    npz_paths: Sequence[Path],
    output_path: Path,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    require_aug: Optional[Tuple[int, str]] = None,
    channel: int = 0,
    cmap: str = "viridis",
) -> str:
    """NPZ パッチを (y_start,x_start) のタイルとして 1 枚の PNG にまとめる。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not npz_paths:
        return ""

    coords = collect_coords(npz_paths, require_aug=require_aug)
    if not coords:
        return ""

    xs_unique = sorted({c[0] for c in coords})
    ys_unique = sorted({c[1] for c in coords})
    cols = len(xs_unique)
    rows = len(ys_unique)

    if vmin is None or vmax is None:
        vals: List[np.ndarray] = []
        for _, _, p in coords:
            d = np.load(p)
            arr = np.asarray(d["data"])
            if arr.ndim == 3:
                arr = arr[channel]
            vals.append(arr[np.isfinite(arr)])
        concatenated = np.concatenate(vals) if len(vals) > 1 else vals[0]
        if vmin is None:
            vmin = float(np.nanmin(concatenated)) if concatenated.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(concatenated)) if concatenated.size else 1.0

    fig_width = max(6, cols * 0.7)
    fig_height = max(4, rows * 0.7)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    x_to_idx = {x: i for i, x in enumerate(xs_unique)}
    y_to_idx = {y: i for i, y in enumerate(ys_unique)}
    used = set()
    for xs, ys, p in coords:
        d = np.load(p)
        arr = np.asarray(d["data"])
        if arr.ndim == 3:
            arr = arr[channel]
        masked = np.ma.masked_invalid(arr)
        ci = x_to_idx[xs]
        ri_bottom = y_to_idx[ys]
        ri = (rows - 1) - ri_bottom
        ax = axes[ri, ci]
        ax.imshow(
            masked,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#333333")
        ax.tick_params(labelbottom=False, labelleft=False)
        used.add((ri, ci))

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in used:
                axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def visualize_all_augs(
    npz_dir: Path,
    out_dir: Path,
    prefix: str,
    *,
    is_2channel: Optional[bool] = None,
    cmap_q: str = "viridis",
    cmap_t: str = "jet",
) -> List[str]:
    """8 通り（rot × flip）それぞれでグリッド PNG を保存。2ch のときチャネル別にも保存。"""
    files = collect_npz_paths(npz_dir)
    if not files:
        return []
    two = is_2channel if is_2channel is not None else detect_two_channel(files)
    outputs: List[str] = []

    if two:
        for channel in range(2):
            channel_name = "Q" if channel == 0 else "Filter"
            cmap = cmap_q if channel == 0 else "gray"
            for rot in (0, 90, 180, 270):
                for flip in ("orig", "lr"):
                    name = f"{prefix}_{channel_name}_grid_rot{rot}_flip{flip}.png"
                    out = visualize_npz_patches(
                        files,
                        out_dir / name,
                        require_aug=(rot, flip),
                        channel=channel,
                        cmap=cmap,
                    )
                    if out:
                        outputs.append(out)
    else:
        for rot in (0, 90, 180, 270):
            for flip in ("orig", "lr"):
                name = f"{prefix}_grid_rot{rot}_flip{flip}.png"
                out = visualize_npz_patches(
                    files,
                    out_dir / name,
                    require_aug=(rot, flip),
                    cmap=cmap_t,
                )
                if out:
                    outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="パッチ NPZ フォルダをグリッド画像（PNG）にまとめて保存する。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="パッチ NPZ が直下にあるディレクトリ、または Q_patches/T_patches を含む親ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG の保存先（省略時は input-dir 配下の viz を使用。複数サブフォルダ時はその下に Q_patches 等）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="単一画像モード時の出力 PNG パス（--all-augs 未指定のとき必須）",
    )
    parser.add_argument(
        "--all-augs",
        action="store_true",
        help="rot0–270 × flip(orig|lr) の 8 枚（2ch 時はチャネル別に追加）を出力",
    )
    parser.add_argument(
        "--prefix",
        default="grid",
        help="出力ファイル名の接頭辞（--all-augs 時）",
    )
    parser.add_argument(
        "--twochannel",
        action="store_true",
        help="データが (C,H,W) の 2ch とみなす（省略時は先頭 NPZ から自動判定）",
    )
    parser.add_argument(
        "--onechannel",
        action="store_true",
        help="データを (H,W) とみなす（2ch 自動判定を上書き）",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="2ch データで単一画像モードのとき参照するチャネル（0=Q, 1=Filter）",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="カラーマップ名（単一画像モード。省略時は viridis）",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"--input-dir がディレクトリではありません: {input_root}")

    patch_dirs = resolve_patch_directories(input_root)
    default_viz = input_root / "viz"
    base_out = args.output_dir.resolve() if args.output_dir else default_viz

    is_2ch: Optional[bool]
    if args.twochannel and args.onechannel:
        raise SystemExit("--twochannel と --onechannel は同時に指定できません。")
    if args.twochannel:
        is_2ch = True
    elif args.onechannel:
        is_2ch = False
    else:
        is_2ch = None

    if args.all_augs:
        all_written: List[str] = []
        for pdir in patch_dirs:
            out_sub = base_out / pdir.name if len(patch_dirs) > 1 else base_out
            out_sub.mkdir(parents=True, exist_ok=True)
            prefix = args.prefix
            two = is_2ch
            if two is None:
                two = detect_two_channel(collect_npz_paths(pdir))
            outs = visualize_all_augs(
                pdir,
                out_sub,
                prefix,
                is_2channel=two,
            )
            all_written.extend(outs)
            for o in outs:
                print("Saved:", o)
        if not all_written:
            print("警告: 保存された PNG がありません（NPZ が空、または座標が取れません）。")
        return

    if args.output is None:
        raise SystemExit("単一画像モードでは --output で PNG パスを指定してください。")

    out_arg = args.output.resolve()
    cmap_default = (args.cmap or "viridis") if args.cmap else None

    for pdir in patch_dirs:
        files = collect_npz_paths(pdir)
        two = is_2ch
        if two is None:
            two = detect_two_channel(files)
        if len(patch_dirs) > 1:
            out_path = out_arg.parent / f"{out_arg.stem}_{pdir.name}{out_arg.suffix}"
        else:
            out_path = out_arg
        if cmap_default is not None:
            cmap_use = cmap_default
        else:
            cmap_use = "viridis" if two else "jet"
        single = visualize_npz_patches(
            files,
            out_path,
            vmin=args.vmin,
            vmax=args.vmax,
            require_aug=None,
            channel=args.channel,
            cmap=cmap_use,
        )
        if single:
            print("Saved:", single)
        else:
            print("警告: 画像を保存できませんでした:", pdir)


if __name__ == "__main__":
    main()
