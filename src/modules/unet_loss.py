"""
U-Net 用の損失: MSE のみ。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """要素平均 MSE。"""
    return F.mse_loss(pred, target)
