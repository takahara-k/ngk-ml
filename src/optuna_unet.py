"""
U-Net のハイパーパラメータ探索（Optuna）。

各 Trial は ``{output_dir}/Trial{N}/`` にベストモデル・推論用ファイル（スケーラー pkl、dataset_info.json）・メタ JSON を保存。
探索終了後 ``{output_dir}/result.json`` に全 Trial 概要とベスト Trial を書き出す。

**Optuna の DB**: 既定では ``{output_dir}/optuna.db``（SQLite）に Study を保存する。再実行時は同じ ``--output-dir`` と ``--study-name`` で続きから再開できる（``load_if_exists=True``）。

# 本番想定（既定の saved/unet_ver0 に optuna.db + Trial フォルダ + result.json）
nohup python src/optuna_unet.py \
    --data-dir saved/ver0/data_preparated \
    --output-dir saved/trained_unet_ver0 \
    --n-trials 20 --epochs 500 \
    --wandb-project unet-heater-optuna \
    --study-name unet_ver0_optuna &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import optuna
from optuna.trial import TrialState

from src.modules.unet_train import run_training
from dotenv import load_dotenv
load_dotenv()
import wandb



def _copy_inference_artifacts(data_dir: Path, trial_dir: Path) -> List[str]:
    """データ準備段階の pkl / json を Trial フォルダへコピー（推論時に同梱できるように）。"""
    names = ["q_scaler.pkl", "t_scaler.pkl", "dataset_info.json"]
    copied: List[str] = []
    for name in names:
        src = data_dir / name
        if src.is_file():
            shutil.copy2(src, trial_dir / name)
            copied.append(name)
        else:
            print(f"警告: {src} が無いためスキップ（学習は続行）")
    return copied


def _save_trial_model_config(
    trial_dir: Path,
    trial: optuna.Trial,
    train_config: Dict[str, Any],
    copied_artifacts: List[str],
) -> None:
    payload = {
        "trial_number": trial.number,
        "optuna_params": trial.params,
        "train_config": {k: v for k, v in train_config.items() if k != "use_wandb"},
        "model": {
            "class": "UNet",
            "input_channels": 2,
            "output_channels": 1,
            "bilinear": train_config.get("bilinear", False),
        },
        "checkpoint_file": "best_model.pt",
        "artifacts_copied": copied_artifacts,
        "notes": "推論時は best_model.pt と q_scaler.pkl / t_scaler.pkl、dataset_info.json を同じディレクトリまたはパス解決で参照する。",
    }
    with open(trial_dir / "inference_bundle.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_single_trial(
    trial: optuna.Trial,
    *,
    data_dir: Path,
    output_root: Path,
    base_epochs: int,
    base_early_stopping: int,
    wandb_project: str,
    study_name: str,
) -> float:
    """1 Trial を実行し、検証損失の最良値（ベスト val）を返す（最小化）。"""
    trial_dir = output_root / f"Trial{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    train_config: Dict[str, Any] = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-5,
        "epochs": base_epochs,
        "early_stopping_patience": base_early_stopping,
        "grad_clip_norm": 1.0,
        "plateau_patience": 10,
        "bilinear": False,
        "use_wandb": True,
        "num_workers": 0,
        "pin_memory": False,
    }

    copied = _copy_inference_artifacts(data_dir, trial_dir)

    run_name = f"{study_name}_Trial{trial.number}"
    summary = run_training(
        train_config,
        data_dir,
        trial_dir,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
    )

    _save_trial_model_config(trial_dir, trial, train_config, copied)

    # Optuna は検証損失を目的にする（過学習しにくい）
    best_val = float(summary.get("best_val_loss", float("inf")))
    trial.set_user_attr("test_mse", summary.get("test_mse"))
    trial.set_user_attr("trial_dir", str(trial_dir))
    return best_val


def default_storage_sqlite_uri(output_dir: Path) -> str:
    """``output_dir/optuna.db`` を指す SQLite URI（絶対パス）。"""
    db_file = (output_dir / "optuna.db").resolve()
    return "sqlite:///" + db_file.as_posix()


def run_optuna_study(
    *,
    data_dir: Path,
    output_dir: Path,
    n_trials: int,
    epochs: int,
    early_stopping_patience: int,
    wandb_project: str,
    study_name: str,
    seed: int,
    storage: Optional[str] = None,
    storage_in_memory: bool = False,
) -> optuna.Study:
    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    wandb.login(key=key)

    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)

    # --storage があれば最優先。なければ --in-memory か既定の {output_dir}/optuna.db
    if storage is not None:
        resolved_storage: Optional[str] = storage
    elif storage_in_memory:
        resolved_storage = None
    else:
        resolved_storage = default_storage_sqlite_uri(output_dir)

    if resolved_storage:
        print(f"Optuna storage: {resolved_storage}")
        study = optuna.create_study(
            study_name=study_name,
            storage=resolved_storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        print("Optuna storage: in-memory（--in-memory）")
        study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        return run_single_trial(
            trial,
            data_dir=data_dir.resolve(),
            output_root=output_dir.resolve(),
            base_epochs=epochs,
            base_early_stopping=early_stopping_patience,
            wandb_project=wandb_project,
            study_name=study_name,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # result.json
    trials_out: List[Dict[str, Any]] = []
    for t in study.trials:
        row: Dict[str, Any] = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "user_attrs": dict(t.user_attrs),
        }
        if t.state == TrialState.FAIL and t.system_attrs.get("fail_reason"):
            row["fail_reason"] = t.system_attrs.get("fail_reason")
        trials_out.append(row)

    complete = [x for x in study.trials if x.state == TrialState.COMPLETE and x.value is not None]
    best_payload: Optional[Dict[str, Any]] = None
    if complete:
        bt = study.best_trial
        best_payload = {
            "number": bt.number,
            "value": bt.value,
            "params": bt.params,
            "user_attrs": dict(bt.user_attrs),
        }

    result = {
        "study_name": study_name,
        "direction": "minimize",
        "metric": "best_val_loss",
        "optuna_storage": resolved_storage if resolved_storage else "in-memory",
        "n_trials_requested": n_trials,
        "n_trials_complete": len(complete),
        "best_trial": best_payload,
        "trials": trials_out,
    }
    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"保存しました: {result_path}")

    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net 用 Optuna ハイパラ探索（W&B）")
    parser.add_argument("--data-dir", type=Path, required=True, help="ml_data_preparation 出力（train_dataset.pt 等）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "saved" / "unet_ver0",
        help="Trial フォルダと result.json の親（既定: saved/unet_ver0）",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100, help="各 Trial のエポック上限")
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--wandb-project", type=str, default="unet-heater-optuna")
    parser.add_argument("--study-name", type=str, default="unet_optuna")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Optuna Study をインメモリのみにする（optuna.db を作らない。スモークテスト向け）",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna DB URI を明示（指定時は --in-memory より優先）。省略時は {output_dir}/optuna.db",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not (data_dir / "train_dataset.pt").is_file():
        raise SystemExit(f"train_dataset.pt がありません: {data_dir}")

    run_optuna_study(
        data_dir=data_dir,
        output_dir=args.output_dir.resolve(),
        n_trials=args.n_trials,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        wandb_project=args.wandb_project,
        study_name=args.study_name,
        seed=args.seed,
        storage=args.storage,
        storage_in_memory=args.in_memory,
    )


if __name__ == "__main__":
    main()
