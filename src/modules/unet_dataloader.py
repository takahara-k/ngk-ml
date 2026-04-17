"""
ml_data_preparation が出力した .pt / dataset_info.json から U-Net 用 DataLoader を構築する。
入力: (N, 2*H*W) の平坦ベクトル → (N, 2, H, W)
ターゲット: (N, H*W) → (N, 1, H, W)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def load_dataset_info(data_dir: Path) -> Dict[str, Any]:
    p = data_dir / "dataset_info.json"
    if not p.is_file():
        raise FileNotFoundError(f"dataset_info.json がありません: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def infer_patch_hw(info: Dict[str, Any]) -> Tuple[int, int]:
    """dataset_info からパッチの (H, W) を取得。"""
    ps = info.get("patch_shape")
    if ps and len(ps) == 2 and ps[0] and ps[1]:
        return int(ps[0]), int(ps[1])
    q0 = info.get("q_channels_flat", {}).get("q_value_elements")
    if q0 is not None:
        s = int(q0**0.5)
        if s * s == q0:
            return s, s
    raise ValueError("patch_shape または q_value_elements からパッチサイズを推定できません。")


class UNetPatchDataset(Dataset):
    """スケール済み平坦テンソルを (C,H,W) に戻して返す。"""

    def __init__(
        self,
        input_flat: torch.Tensor,
        target_flat: torch.Tensor,
        patch_h: int,
        patch_w: int,
        q_channels: int = 2,
    ) -> None:
        if input_flat.shape[0] != target_flat.shape[0]:
            raise ValueError("input と target のサンプル数が一致しません。")
        expected_in = q_channels * patch_h * patch_w
        if input_flat.dim() != 2 or input_flat.shape[1] != expected_in:
            raise ValueError(
                f"入力次元の不一致: 期待 {expected_in}, 実際 {tuple(input_flat.shape)}"
            )
        t_el = patch_h * patch_w
        if target_flat.dim() != 2 or target_flat.shape[1] != t_el:
            raise ValueError(
                f"ターゲット次元の不一致: 期待 {t_el}, 実際 {tuple(target_flat.shape)}"
            )
        self.input_flat = input_flat
        self.target_flat = target_flat
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.q_channels = q_channels

    def __len__(self) -> int:
        return self.input_flat.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_flat[idx].view(self.q_channels, self.patch_h, self.patch_w)
        y = self.target_flat[idx].view(1, self.patch_h, self.patch_w)
        return x, y


def create_unet_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    q_channels: int = 2,
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """
    data_dir に train_dataset.pt / val_dataset.pt / test_dataset.pt と dataset_info.json がある前提。

    Returns:
        loaders: keys 'train', 'val', 'test'
        meta: patch_hw, dataset_info の抜粋
    """
    root = Path(data_dir).resolve()
    info = load_dataset_info(root)
    ph, pw = infer_patch_hw(info)

    def _load_split(name: str) -> torch.Tensor:
        p = root / f"{name}_dataset.pt"
        if not p.is_file():
            raise FileNotFoundError(p)
        blob = torch.load(p, map_location="cpu")
        return blob["input"], blob["target"]

    train_in, train_t = _load_split("train")
    val_in, val_t = _load_split("val")
    test_in, test_t = _load_split("test")

    train_ds = UNetPatchDataset(train_in, train_t, ph, pw, q_channels=q_channels)
    val_ds = UNetPatchDataset(val_in, val_t, ph, pw, q_channels=q_channels)
    test_ds = UNetPatchDataset(test_in, test_t, ph, pw, q_channels=q_channels)

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    meta = {
        "patch_h": ph,
        "patch_w": pw,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "dataset_info": info,
    }
    return loaders, meta
