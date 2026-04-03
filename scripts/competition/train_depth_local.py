from __future__ import annotations

import dataclasses
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tyro

from openpi.training import config as config_lib

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_lib  # noqa: E402


@dataclasses.dataclass
class Args:
    exp_name: str
    config_name: str = "acot_icra_simulation_challenge_reasoning_to_action_clean_depth_pi05libero_continue_45000"
    checkpoint_base_dir: str = "/storage/nobackup/yiduo/acotvla_depth_checkpoints"
    num_train_steps: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    log_interval: int | None = None
    val_split_ratio: float | None = None
    val_interval: int | None = None
    val_num_batches: int | None = None
    save_interval: int | None = None
    keep_period: int | None = None
    wandb_enabled: bool = False
    overwrite: bool = False
    resume: bool = False
    fsdp_devices: int | None = None
    seed: int | None = None


def build_config(args: Args):
    base = config_lib.get_config(args.config_name)
    replace_kwargs = {
        "exp_name": args.exp_name,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "wandb_enabled": args.wandb_enabled,
        "overwrite": args.overwrite,
        "resume": args.resume,
    }

    optional_fields = {
        "num_train_steps": args.num_train_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "log_interval": args.log_interval,
        "val_split_ratio": args.val_split_ratio,
        "val_interval": args.val_interval,
        "val_num_batches": args.val_num_batches,
        "save_interval": args.save_interval,
        "keep_period": args.keep_period,
        "fsdp_devices": args.fsdp_devices,
        "seed": args.seed,
    }
    replace_kwargs.update({k: v for k, v in optional_fields.items() if v is not None})
    return dataclasses.replace(base, **replace_kwargs)


def main(args: Args) -> None:
    config = build_config(args)
    train_lib.main(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
