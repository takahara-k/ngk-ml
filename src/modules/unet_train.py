"""
U-Net 学習・評価。ログは Weights & Biases（W&B）。損失曲線は W&B のメトリクスで可視化（ローカル PNG は保存しない）。
損失は MSE のみ。ハイパーパラメータは元スクリプトのデフォルトに準拠。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import wandb

from src.models.UNET import create_unet_model
from src.modules.unet_dataloader import create_unet_dataloaders
from src.modules.unet_loss import mse_loss


def log_line(msg: str, *, use_wandb: bool) -> None:
    """標準出力と W&B Run コンソールの両方へ同じ行を出す。"""
    print(msg, flush=True)
    if use_wandb:
        wandb.termlog(msg)


def describe_compute_device(device: torch.device) -> str:
    """人間可読な 1 行（ログ用）。"""
    if device.type == "cuda":
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"CUDA (gpu={idx}, name={name})"
        except Exception as exc:  # noqa: BLE001
            return f"CUDA (詳細取得失敗: {exc})"
    return "CPU"


class UNetTrainer:
    """U-Net 学習（W&B ログ）。損失は常に MSE。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[compute] 計算デバイス: {describe_compute_device(self.device)}")
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.meta: Dict[str, Any] = {}

    def load_data(self, data_dir: str | Path) -> None:
        loaders, meta = create_unet_dataloaders(
            data_dir,
            batch_size=int(self.config["batch_size"]),
            num_workers=int(self.config.get("num_workers", 0)),
            pin_memory=bool(self.config.get("pin_memory", False) and torch.cuda.is_available()),
        )
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        self.meta = meta
        print(f"訓練: {meta['train_size']} 検証: {meta['val_size']} テスト: {meta['test_size']}")
        print(f"パッチ: {meta['patch_h']} x {meta['patch_w']}")

    def setup_model(self) -> None:
        self.model = create_unet_model(
            input_channels=2,
            output_channels=1,
            bilinear=bool(self.config.get("bilinear", False)),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=int(self.config.get("plateau_patience", 10)),
            factor=0.5,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"モデルパラメータ数: {n_params:,} 損失: MSE  scheduler: ReduceLROnPlateau  "
            f"({describe_compute_device(self.device)})"
        )

    def train_epoch(self) -> Dict[str, float]:
        assert self.model and self.train_loader and self.optimizer
        self.model.train()
        total = 0.0
        n_batches = 0
        for q_data, t_data in self.train_loader:
            q_data = q_data.to(self.device)
            t_data = t_data.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(q_data)
            loss = mse_loss(out, t_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip_norm"]))
            self.optimizer.step()
            total += loss.item()
            n_batches += 1
        return {"train_loss": total / max(n_batches, 1)}

    def validate_epoch(self) -> Dict[str, float]:
        assert self.model and self.val_loader
        self.model.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for q_data, t_data in self.val_loader:
                q_data = q_data.to(self.device)
                t_data = t_data.to(self.device)
                out = self.model(q_data)
                loss = mse_loss(out, t_data)
                total += loss.item()
                n_batches += 1
        return {"val_loss": total / max(n_batches, 1)}

    def train(self) -> Dict[str, Any]:
        assert self.model and self.train_loader and self.val_loader
        epochs = int(self.config["epochs"])
        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val = float("inf")
        patience = int(self.config.get("early_stopping_patience", 0) or 0)
        patience_cnt = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(epochs):
            tr = self.train_epoch()
            va = self.validate_epoch()
            train_losses.append(tr["train_loss"])
            val_losses.append(va["val_loss"])

            assert self.scheduler is not None
            self.scheduler.step(va["val_loss"])

            lr = self.optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
            uw = bool(self.config.get("use_wandb", True))
            epoch_msg = (
                f"Epoch {epoch + 1}/{epochs} train={tr['train_loss']:.6f} "
                f"val={va['val_loss']:.6f} lr={lr:.2e}"
            )
            log_line(epoch_msg, use_wandb=uw)

            if uw:
                wandb.log(
                    {"train_loss": tr["train_loss"], "val_loss": va["val_loss"], "learning_rate": lr},
                    step=epoch,
                )

            if va["val_loss"] < best_val:
                best_val = va["val_loss"]
                patience_cnt = 0
                assert self.model is not None
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1

            if patience > 0 and patience_cnt >= patience:
                log_line(f"Early stopping at epoch {epoch + 1}", use_wandb=uw)
                break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
        }

    def evaluate_loader(self, loader: DataLoader, split_name: str = "test") -> Dict[str, float]:
        """MSE（平均）のみ。"""
        assert self.model is not None
        self.model.eval()
        total_mse = 0.0
        n_batches = 0
        with torch.no_grad():
            for q_data, t_data in loader:
                q_data = q_data.to(self.device)
                t_data = t_data.to(self.device)
                out = self.model(q_data)
                total_mse += mse_loss(out, t_data).item()
                n_batches += 1
        n = max(n_batches, 1)
        return {f"{split_name}_mse": total_mse / n}

    def evaluate_test(self) -> Dict[str, float]:
        assert self.test_loader is not None
        return self.evaluate_loader(self.test_loader, "test")


def run_training(
    config: Dict[str, Any],
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    wandb_project: str,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    学習〜評価を一括実行。ベスト重みを ``output_dir / best_model.pt`` に保存。
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if config.get("use_wandb", True):
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        wandb.config.update(config)

    trainer = UNetTrainer(config)
    if config.get("use_wandb", True):
        wandb.config.update(
            {
                "compute_device": describe_compute_device(trainer.device),
                "torch_device": str(trainer.device),
            }
        )

    trainer.load_data(data_dir)
    trainer.setup_model()
    results = trainer.train()
    test_metrics = trainer.evaluate_test()

    uw = bool(config.get("use_wandb", True))
    if uw:
        wandb.log({k: v for k, v in test_metrics.items()})

    assert trainer.model is not None
    torch.save(trainer.model.state_dict(), out / "best_model.pt")
    summary = {
        "best_val_loss": results["best_val_loss"],
        **test_metrics,
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
    }
    with open(out / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in summary.items() if k not in ("train_losses", "val_losses")},
            f,
            indent=2,
            ensure_ascii=False,
        )

    log_line(f"テスト指標: {test_metrics}", use_wandb=uw)
    if uw:
        wandb.finish()
    return summary


def evaluate_only(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """保存済み重みで train/val/test を評価（W&B なし）。"""
    cfg = dict(config or {})
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("grad_clip_norm", 1.0)
    cfg.setdefault("plateau_patience", 10)
    cfg.setdefault("bilinear", False)
    cfg["use_wandb"] = False
    trainer = UNetTrainer(cfg)
    trainer.load_data(data_dir)
    trainer.setup_model()
    sd = torch.load(checkpoint_path, map_location=trainer.device)
    assert trainer.model is not None
    trainer.model.load_state_dict(sd)

    out: Dict[str, float] = {}
    for name, loader in [
        ("train", trainer.train_loader),
        ("val", trainer.val_loader),
        ("test", trainer.test_loader),
    ]:
        assert loader is not None
        m = trainer.evaluate_loader(loader, name)
        out.update(m)
    return out


def main() -> None:
    # 元スクリプトの default_training に合わせる
    parser = argparse.ArgumentParser(description="U-Net 学習（W&B / MSE）")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", type=str, default="unet-heater")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    args = parser.parse_args()

    config: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "grad_clip_norm": 1.0,
        "plateau_patience": 10,
        "bilinear": False,
        "use_wandb": not args.no_wandb,
        "num_workers": 0,
        "pin_memory": False,
    }

    run_training(
        config,
        args.data_dir,
        args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
